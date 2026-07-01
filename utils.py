"""
utils.py
--------
Shared utilities for LectureScribe:
- Environment variable loading
- Audio extraction from local video or YouTube URL
- SRT/subtitle parsing
- Transcript chunking
- Folder processing
"""

import os
import re
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from subtitle_parser import load_subtitle, find_subtitle_for_video, can_parse

load_dotenv()


def get_env(key: str) -> str:
    """Load a required environment variable or exit with a clear error."""
    value = os.getenv(key)
    if not value:
        print(f"[ERROR] Missing environment variable: {key}. Check your .env file.")
        exit(1)
    return value


# ── Audio extraction ───────────────────────────────────────────────────────────

def extract_audio(input_path: str, job=None, output_dir: str = "outputs") -> str:
    """
    Extract audio from a local video file or YouTube URL.
    If a Job is provided, saves to job.audio_dir/audio.wav.
    Returns the path to a 16kHz mono WAV file.
    """
    if job is not None:
        out_wav = str(job.audio_dir / "audio.wav")
    else:
        os.makedirs(output_dir, exist_ok=True)
        out_wav = str(Path(output_dir) / "audio.wav")

    is_url = input_path.startswith("http")
    if is_url:
        _extract_from_youtube(input_path, out_wav)
    else:
        _extract_from_local(input_path, out_wav)

    return out_wav


def _extract_from_youtube(url: str, output_wav: str) -> None:
    """Download audio from YouTube using yt-dlp and convert to 16kHz mono WAV."""
    temp_audio = output_wav.replace(".wav", ".%(ext)s")

    result = subprocess.run(["yt-dlp", "--version"], capture_output=True)
    if result.returncode != 0:
        print("[ERROR] yt-dlp not found. Run: pip install yt-dlp")
        exit(1)

    print("  Downloading audio from YouTube...")
    dl_result = subprocess.run([
        "yt-dlp", "-x",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", temp_audio,
        url
    ], capture_output=True, text=True)

    if dl_result.returncode != 0:
        print(f"[ERROR] Could not download video: {dl_result.stderr.strip()}")
        exit(1)

    candidate = output_wav
    if not os.path.exists(candidate):
        out_dir = str(Path(output_wav).parent)
        wavs = list(Path(out_dir).glob("*.wav"))
        if not wavs:
            print("[ERROR] yt-dlp finished but no WAV file found.")
            exit(1)
        candidate = str(wavs[0])

    _resample_wav(candidate, output_wav)


def _extract_from_local(input_path: str, output_wav: str) -> None:
    """Extract audio from a local video file using ffmpeg."""
    if not os.path.exists(input_path):
        print(f"[ERROR] File not found: {input_path}")
        exit(1)

    result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    if result.returncode != 0:
        print("[ERROR] ffmpeg not found. Install from https://ffmpeg.org and add to PATH.")
        exit(1)

    print("  Extracting audio from local file...")
    ff_result = subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "16000", "-ac", "1",
        output_wav
    ], capture_output=True, text=True)

    if ff_result.returncode != 0:
        print(f"[ERROR] ffmpeg failed: {ff_result.stderr.strip()}")
        exit(1)


def _resample_wav(input_wav: str, output_wav: str) -> None:
    """Resample any WAV to 16kHz mono using ffmpeg."""
    if input_wav == output_wav:
        temp = output_wav + ".tmp.wav"
        subprocess.run([
            "ffmpeg", "-y", "-i", input_wav,
            "-ar", "16000", "-ac", "1", temp
        ], capture_output=True)
        os.replace(temp, output_wav)
    else:
        subprocess.run([
            "ffmpeg", "-y", "-i", input_wav,
            "-ar", "16000", "-ac", "1", output_wav
        ], capture_output=True)
        if os.path.exists(input_wav) and input_wav != output_wav:
            os.remove(input_wav)


# ── Transcript chunking ────────────────────────────────────────────────────────

def chunk_transcript(transcript: str, max_words: int = 3000) -> list[str]:
    """
    Split a long transcript into chunks of max_words words.
    Splits on sentence boundaries where possible.
    Falls back to hard split if no sentence boundary is found.
    """
    words = transcript.split()
    if len(words) <= max_words:
        return [transcript]

    chunks = []
    current_words = []

    for word in words:
        current_words.append(word)

        if len(current_words) >= max_words and word.endswith(('.', '?', '!')):
            chunks.append(' '.join(current_words))
            current_words = []
        elif len(current_words) >= int(max_words * 1.2):
            chunks.append(' '.join(current_words))
            current_words = []

    if current_words:
        chunks.append(' '.join(current_words))

    return chunks


# ── Subtitle parsing ───────────────────────────────────────────────────────────

def parse_srt(srt_path: str) -> str:
    """Backward-compatible wrapper. Use load_subtitle() for new code."""
    return load_subtitle(srt_path)


def find_srt_for_video(video_path: str) -> str | None:
    """Backward-compatible wrapper. Use find_subtitle_for_video() for new code."""
    return find_subtitle_for_video(video_path)


# ── Folder processing ──────────────────────────────────────────────────────────

def get_folder_transcript(folder_path: str, use_srt: bool = True, job=None) -> tuple[str, str]:
    """
    Given a folder, find all video files in sorted order.
    For each video: use SRT if available and use_srt=True, else Whisper.
    If a Job is provided, saves individual transcripts to job.trans_dir.
    Returns (combined_transcript, folder_title).
    """
    from transcriber import transcribe as whisper_transcribe

    folder = Path(folder_path)
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    video_files = sorted([f for f in folder.iterdir()
                          if f.suffix.lower() in video_exts])

    if not video_files:
        print(f"[ERROR] No video files found in: {folder_path}")
        exit(1)

    print(f"  Found {len(video_files)} video(s) in folder.")
    parts = []

    for vf in video_files:
        stem = vf.stem

        if use_srt:
            sub_path = find_subtitle_for_video(str(vf))
            if sub_path:
                ext = Path(sub_path).suffix.lower()
                print(f"  [{ext[1:].upper()}]     {vf.name}")
                text = load_subtitle(sub_path)
            else:
                print(f"  [Whisper] {vf.name} (no subtitle found)")
                
                if job:
                    audio = extract_audio(str(vf), job=job)
                    out_wav = str(job.audio_dir / f"{stem}.wav")
                    # rename generic audio.wav to per-lecture name
                    if os.path.exists(audio) and audio != out_wav:
                        os.replace(audio, out_wav)
                    audio = out_wav
                else:
                    audio = extract_audio(str(vf))
                text = whisper_transcribe(audio)
        else:
            print(f"  [Whisper] {vf.name}")
            if job:
                out_wav = str(job.audio_dir / f"{stem}.wav")
                _extract_from_local(str(vf), out_wav)
                audio = out_wav
            else:
                audio = extract_audio(str(vf))
            text = whisper_transcribe(audio)

        # Save per-lecture transcript to job workspace
        if job:
            trans_file = job.transcript_path(stem)
            with open(trans_file, "w", encoding="utf-8") as f:
                f.write(text)

        parts.append(text)

    combined = " ".join(parts)

    # Save merged transcript
    if job:
        with open(job.merged_transcript, "w", encoding="utf-8") as f:
            f.write(combined)

    return combined, folder.name
