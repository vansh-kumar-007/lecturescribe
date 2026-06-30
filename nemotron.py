"""
nemotron.py
-----------
Handles communication with the NVIDIA Nemotron-Ultra API.
Sends transcript chunks and returns structured JSON notes data.
Includes retry logic for invalid JSON responses.
"""

import json
import os
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
        # Only collect content, skip reasoning_content
        if delta.content is not None:
            content += delta.content

    return content.strip()


def parse_json_response(raw: str, attempt: int = 1) -> dict:
    """
    Parse Nemotron's response as JSON.
    Retries once with a stricter prompt if parsing fails.
    """
    # Strip markdown fences if model added them anyway
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


def process_chunk(client: OpenAI, chunk_text: str, chunk_index: int) -> dict:
    """
    Send one transcript chunk to Nemotron and return parsed JSON.
    Retries once with explicit JSON reminder if parsing fails.
    """
    system_prompt, user_prompt = build_prompt(chunk_text)

    print(f"    Sending chunk {chunk_index + 1} to Nemotron...")
    raw = call_nemotron(client, system_prompt, user_prompt)

    try:
        return parse_json_response(raw, attempt=1)
    except ValueError as e:
        if str(e).startswith("JSON_PARSE_FAILED:"):
            print(f"    [WARN] Chunk {chunk_index + 1} returned invalid JSON. Retrying...")
            # Append explicit reminder and retry
            retry_prompt = user_prompt + (
                "\n\nYour previous response was not valid JSON. "
                "Return ONLY the JSON object, nothing else. Start with { and end with }."
            )
            raw2 = call_nemotron(client, system_prompt, retry_prompt)
            try:
                return parse_json_response(raw2, attempt=2)
            except ValueError:
                # Save debug output and exit
                debug_path = "outputs/debug_raw.txt"
                with open(debug_path, "w", encoding="utf-8") as f:
                    f.write(raw2)
                print(f"[ERROR] Nemotron returned invalid JSON after retry. Raw response saved to {debug_path}")
                exit(1)
        raise


def merge_chunks(results: list[dict]) -> dict:
    """
    Merge structured JSON results from multiple transcript chunks.
    Metadata comes from chunk 1. Blocks are concatenated in order.
    Post-processes consecutive duplicate headings and split sentences.
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
    """
    Clean up merged blocks:
    - Remove consecutive duplicate headings
    - Merge body blocks that appear to continue a split sentence
    """
    if not blocks:
        return blocks

    cleaned = [blocks[0]]

    for i in range(1, len(blocks)):
        prev = cleaned[-1]
        curr = blocks[i]

        # Drop duplicate consecutive headings
        if (curr["type"] == "heading"
                and prev["type"] == "heading"
                and curr["content"].strip().lower() == prev["content"].strip().lower()):
            continue

        # Merge body block that continues a split sentence
        if (curr["type"] == "body"
                and prev["type"] == "body"
                and prev["content"].strip()
                and not prev["content"].strip().endswith(('.', '?', '!'))):
            cleaned[-1]["content"] = prev["content"].strip() + " " + curr["content"].strip()
            continue

        cleaned.append(curr)

    return cleaned


def analyze_transcript(transcript: str) -> dict:
    """
    Full pipeline: chunk transcript → call Nemotron per chunk → merge results.
    Returns the final merged structured notes dict.
    """
    client = get_client()
    chunks = chunk_transcript(transcript)
    print(f"  Transcript split into {len(chunks)} chunk(s).")

    results = []
    for i, chunk in enumerate(chunks):
        result = process_chunk(client, chunk, i)
        results.append(result)

    return merge_chunks(results)