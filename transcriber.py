"""
transcriber.py
--------------
Handles speech-to-text transcription using OpenAI Whisper large-v3.
Runs locally on GPU (CUDA). Falls back to 'medium' model on VRAM OOM.

Phase 3: Segmented transcription pipeline.
- Splits audio into 30-second segments using ffmpeg
- Transcribes each segment individually
- Saves each segment transcript immediately to disk
- Resumes from last completed segment if interrupted
- Progress bar via rich

The old transcribe() function is kept for backward compatibility
(used by Kaggle notebook and URL mode without a Job).
"""

import os
import re
import math
import warnings
import subprocess
import tempfile
from pathlib import Path

import torch
import whisper
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn


# ── Constants ──────────────────────────────────────────────────────────────────
SEGMENT_DURATION = 30   # seconds per audio segment
DEFAULT_MODEL    = "large-v3"


# ── Public API ─────────────────────────────────────────────────────────────────

def transcribe(audio_path: str, model_name: str = DEFAULT_MODEL) -> str:
    """
    Simple transcription — no job, no resume, no segmentation.
    Used for URL mode and Kaggle notebook.
    Falls back to 'medium' if CUDA OOM.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading Whisper {model_name} on {device.upper()}...")

    try:
        model = _load_model(model_name, device)
        return _run_full_transcription(model, audio_path, model_name, device)

    except torch.cuda.OutOfMemoryError:
        print(f"  [WARN] VRAM insufficient for {model_name}. Retrying with 'medium'...")
        torch.cuda.empty_cache()
        model_name = "medium"
        model = _load_model(model_name, device)
        return _run_full_transcription(model, audio_path, model_name, device)


def transcribe_segmented(audio_path: str, job, model_name: str = DEFAULT_MODEL) -> str:
    """
    Segmented transcription pipeline with resume support.

    Steps:
      1. Get audio duration
      2. Split into 30-second segments (ffmpeg)
      3. For each segment:
         - Check if transcript already exists → skip (resume)
         - Otherwise transcribe with Whisper → save immediately
      4. Merge all segment transcripts
      5. Save merged transcript to job.merged_transcript

    Returns the full merged transcript as a string.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger = job.get_logger()

    print(f"  Loading Whisper {model_name} on {device.upper()}...")
    logger.info(f"Segmented transcription started: {audio_path}")

    try:
        model = _load_model(model_name, device)
    except torch.cuda.OutOfMemoryError:
        print(f"  [WARN] VRAM insufficient for {model_name}. Retrying with 'medium'...")
        torch.cuda.empty_cache()
        model_name = "medium"
        model = _load_model(model_name, device)

    # Get audio duration
    duration = _get_audio_duration(audio_path)
    n_segments = math.ceil(duration / SEGMENT_DURATION)
    print(f"  Audio duration: {duration:.1f}s → {n_segments} segments of {SEGMENT_DURATION}s each")
    logger.info(f"Duration: {duration:.1f}s, segments: {n_segments}")

    # Segment directory inside job transcripts folder
    seg_dir = job.trans_dir / "segments"
    seg_dir.mkdir(exist_ok=True)

    # Count already-done segments for resume display
    done_count = sum(1 for i in range(n_segments)
                     if (seg_dir / f"seg_{i:05d}.txt").exists())
    if done_count > 0:
        print(f"  Resuming: {done_count}/{n_segments} segments already transcribed.")

    # Transcribe segments with progress bar
    with Progress(
        TextColumn("  [bold cyan]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            f"Whisper {model_name}",
            total=n_segments,
            completed=done_count
        )

        for i in range(n_segments):
            seg_txt = seg_dir / f"seg_{i:05d}.txt"

            # Resume: skip completed segments
            if seg_txt.exists():
                progress.advance(task)
                continue

            start_sec = i * SEGMENT_DURATION

            # Extract segment audio to temp file
            seg_wav = _extract_segment(audio_path, start_sec, SEGMENT_DURATION)

            try:
                text = _transcribe_segment(model, seg_wav, device)
            finally:
                if os.path.exists(seg_wav):
                    os.remove(seg_wav)

            # Save immediately
            seg_txt.write_text(text, encoding="utf-8")

            # Update state
            job.write_state(
                current_step="transcription",
                segments_done=i + 1,
                segments_total=n_segments,
                status="running"
            )
            logger.debug(f"Segment {i+1}/{n_segments} done: {len(text.split())} words")

            progress.advance(task)

    # Merge all segments
    texts = []
    for i in range(n_segments):
        seg_txt = seg_dir / f"seg_{i:05d}.txt"
        if seg_txt.exists():
            texts.append(seg_txt.read_text(encoding="utf-8").strip())

    merged = " ".join(t for t in texts if t)
    word_count = len(merged.split())
    print(f"  Transcription complete. ~{word_count} words.")
    logger.info(f"Transcription complete: {word_count} words")

    # Save merged transcript
    job.merged_transcript.write_text(merged, encoding="utf-8")

    return merged


# ── Internal helpers ───────────────────────────────────────────────────────────

def _load_model(model_name: str, device: str):
    """Load a Whisper model onto the specified device."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return whisper.load_model(model_name, device=device)


def _run_full_transcription(model, audio_path: str, model_name: str, device: str) -> str:
    """Run a single Whisper transcription call on the full audio file."""
    print(f"  Transcribing with Whisper {model_name}... (this may take a while)")

    result = model.transcribe(
        audio_path,
        fp16=(device == "cuda"),
        verbose=False
    )

    text = result["text"].strip()
    print(f"  Transcription complete. ~{len(text.split())} words.")
    return text


def _transcribe_segment(model, seg_wav: str, device: str) -> str:
    """Transcribe a single audio segment and return plain text."""
    import contextlib, io
    # Suppress Whisper's per-segment tqdm output so our rich bar is clean
    f = io.StringIO()
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        result = model.transcribe(
            seg_wav,
            fp16=(device == "cuda"),
            verbose=False
        )
    return result["text"].strip()


def _get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        audio_path
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")

    return float(result.stdout.strip())


def _extract_segment(audio_path: str, start_sec: float, duration: float) -> str:
    """
    Extract a audio segment using ffmpeg.
    Returns path to a temporary WAV file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()

    result = subprocess.run([
        "ffmpeg", "-y",
        "-ss", str(start_sec),
        "-t", str(duration),
        "-i", audio_path,
        "-ar", "16000", "-ac", "1",
        "-acodec", "pcm_s16le",
        tmp.name
    ], capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg segment extraction failed: {result.stderr.strip()}")

    return tmp.name
