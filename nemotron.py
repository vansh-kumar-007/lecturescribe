"""
nemotron.py
-----------
Handles communication with the NVIDIA Nemotron-Ultra API.
Sends transcript chunks and returns structured JSON notes data.
Includes retry logic for rate limits and invalid JSON responses.
If a Job is provided, saves per-chunk results to job.ai_dir for resume support.
"""

import json
import os
import time
from openai import OpenAI
from utils import get_env, chunk_transcript
from prompts import build_prompt


def get_client() -> OpenAI:
    """Initialize the NVIDIA Nemotron API client."""
    return OpenAI(
        api_key=get_env("NVIDIA_API_KEY"),
        base_url="https://integrate.api.nvidia.com/v1"
    )


def call_nemotron(client: OpenAI, system_prompt: str, user_prompt: str) -> str:
    """
    Make a single streaming call to Nemotron.
    Returns the full response text (reasoning stripped, content only).
    """
    completion = client.chat.completions.create(
        model="nvidia/nemotron-3-ultra-550b-a55b",
        messages=[{"role": "user", "content": user_prompt}],
        temperature=1,
        top_p=0.95,
        max_tokens=16384,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_budget": 8192
        },
        stream=True
    )

    content = ""
    for chunk in completion:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content is not None:
            content += delta.content

    return content.strip()


def parse_json_response(raw: str, attempt: int = 1) -> dict:
    """Parse Nemotron's response as JSON. Strips markdown fences if present."""
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
        clean = clean.strip()

    try:
        return json.loads(clean)
    except json.JSONDecodeError as e:
        if attempt == 1:
            raise ValueError(f"JSON_PARSE_FAILED:{raw}")
        raise ValueError(f"[ERROR] Nemotron returned invalid JSON after retry: {e}")


def process_chunk(client: OpenAI, chunk_text: str, chunk_index: int, job=None) -> dict:
    """
    Send one transcript chunk to Nemotron and return parsed JSON.
    If a Job is provided, saves prompt and result to job workspace.
    Retries up to 3 times with exponential backoff on rate limits.
    """
    system_prompt, user_prompt = build_prompt(chunk_text)
    max_retries = 3

    # Save prompt for debugging
    if job:
        prompt_file = job.prompt_path(chunk_index)
        with open(prompt_file, "w", encoding="utf-8") as f:
            f.write(f"=== SYSTEM ===\n{system_prompt}\n\n=== USER ===\n{user_prompt}")

    for attempt in range(max_retries):
        try:
            print(f"    Sending chunk {chunk_index + 1} to Nemotron...")
            raw = call_nemotron(client, system_prompt, user_prompt)

            try:
                result = parse_json_response(raw, attempt=1)
            except ValueError as e:
                if str(e).startswith("JSON_PARSE_FAILED:"):
                    print(f"    [WARN] Chunk {chunk_index + 1} returned invalid JSON. Retrying...")
                    retry_prompt = user_prompt + (
                        "\n\nYour previous response was not valid JSON. "
                        "Return ONLY the JSON object, nothing else. Start with { and end with }."
                    )
                    raw2 = call_nemotron(client, system_prompt, retry_prompt)
                    try:
                        result = parse_json_response(raw2, attempt=2)
                    except ValueError:
                        debug_path = str(job.ai_dir / "debug_raw.txt") if job else "outputs/debug_raw.txt"
                        with open(debug_path, "w", encoding="utf-8") as f:
                            f.write(raw2)
                        print(f"[ERROR] Invalid JSON after retry. Raw saved to {debug_path}")
                        exit(1)
                else:
                    raise

            # Save chunk result to job workspace
            if job:
                chunk_file = job.ai_chunk_path(chunk_index)
                with open(chunk_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)

            return result

        except Exception as e:
            err_str = str(e)
            if "ResourceExhausted" in err_str or "rate" in err_str.lower() or "limit" in err_str.lower():
                wait = 60 * (attempt + 1)
                print(f"    [WARN] Rate limit hit on chunk {chunk_index + 1}. Waiting {wait}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait)
                continue
            raise

    print(f"[ERROR] Chunk {chunk_index + 1} failed after {max_retries} retries.")
    exit(1)


def merge_chunks(results: list[dict]) -> dict:
    """
    Merge structured JSON results from multiple transcript chunks.
    Metadata comes from chunk 1. Blocks are concatenated in order.
    """
    if len(results) == 1:
        return results[0]

    merged = {
        "subject": results[0].get("subject", ""),
        "lecture_title": results[0].get("lecture_title", ""),
        "topics_covered": results[0].get("topics_covered", []),
        "estimated_read_time_minutes": sum(r.get("estimated_read_time_minutes", 0) for r in results),
        "blocks": []
    }

    for result in results:
        merged["blocks"].extend(result.get("blocks", []))

    merged["blocks"] = _post_process_blocks(merged["blocks"])
    return merged


def _post_process_blocks(blocks: list[dict]) -> list[dict]:
    """Remove duplicate consecutive headings and merge split body sentences."""
    if not blocks:
        return blocks

    cleaned = [blocks[0]]

    for i in range(1, len(blocks)):
        prev = cleaned[-1]
        curr = blocks[i]

        if (curr["type"] == "heading"
                and prev["type"] == "heading"
                and curr["content"].strip().lower() == prev["content"].strip().lower()):
            continue

        if (curr["type"] == "body"
                and prev["type"] == "body"
                and prev["content"].strip()
                and not prev["content"].strip().endswith(('.', '?', '!'))):
            cleaned[-1]["content"] = prev["content"].strip() + " " + curr["content"].strip()
            continue

        cleaned.append(curr)

    return cleaned


def analyze_transcript(transcript: str, job=None) -> dict:
    """
    Full pipeline: chunk transcript → call Nemotron per chunk → merge results.
    If a Job is provided:
      - Saves each chunk result to job.ai_dir/chunk<N>.json
      - Skips chunks that already have a saved result (resume support)
      - Saves merged result to job.merged_ai
    """
    client = get_client()
    chunks = chunk_transcript(transcript)
    print(f"  Transcript split into {len(chunks)} chunk(s).")

    results = []
    for i, chunk in enumerate(chunks):
        # Resume: skip chunks already processed
        if job:
            chunk_file = job.ai_chunk_path(i)
            if chunk_file.exists():
                print(f"    Chunk {i + 1} already processed. Loading from disk...")
                with open(chunk_file, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
                continue

        result = process_chunk(client, chunk, i, job=job)
        results.append(result)

        if i < len(chunks) - 1:
            print(f"    Pausing 15s before next chunk...")
            time.sleep(15)

    merged = merge_chunks(results)

    # Save merged result
    if job:
        with open(job.merged_ai, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)

    return merged
