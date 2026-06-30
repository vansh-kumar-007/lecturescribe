"""
utils.py
--------
Shared utilities for LectureScribe:
- Environment variable loading
- Audio extraction from local MP4 or YouTube URL (output: 16kHz mono WAV)
"""

import os
import subprocess
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_env(key: str) -> str:
    """Load a required environment variable or exit with a clear error."""
    value = os.getenv(key)
    if not value:
        print(f"[ERROR] Missing environment variable: {key}. Check your .env file.")
        exit(1)
    return value


def extract_audio(input_path: str, output_dir: str = "outputs") -> str:
    """
    Extract audio from a local video file or YouTube URL.
    Returns the path to a 16kHz mono WAV file.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_wav = str(Path(output_dir) / "audio.wav")

    is_url = input_path.startswith("http")

    if is_url:
        _extract_from_youtube(input_path, output_wav)
    else:
        _extract_from_local(input_path, output_wav)

    return output_wav


def _extract_from_youtube(url: str, output_wav: str) -> None:
    """Download audio from YouTube using yt-dlp and convert to 16kHz mono WAV."""
    temp_audio = output_wav.replace(".wav", ".%(ext)s")

    # Check yt-dlp is available
    result = subprocess.run(["yt-dlp", "--version"], capture_output=True)
    if result.returncode != 0:
        print("[ERROR] yt-dlp not found. Run: pip install yt-dlp")
        exit(1)

    print("  Downloading audio from YouTube...")
    dl_result = subprocess.run([
        "yt-dlp",
        "-x",                          # extract audio only
        "--audio-format", "wav",
        "--audio-quality", "0",
        "-o", temp_audio,
        url
    ], capture_output=True, text=True)

    if dl_result.returncode != 0:
        print(f"[ERROR] Could not download video: {dl_result.stderr.strip()}")
        exit(1)

    # yt-dlp may output a different filename — find it
    candidate = output_wav.replace(".wav", ".wav")
    if not os.path.exists(candidate):
        # Search outputs folder for any wav
        wavs = list(Path("outputs").glob("*.wav"))
        if not wavs:
            print("[ERROR] yt-dlp finished but no WAV file found in outputs/")
            exit(1)
        candidate = str(wavs[0])

    # Resample to 16kHz mono
    _resample_wav(candidate, output_wav)


def _extract_from_local(input_path: str, output_wav: str) -> None:
    """Extract audio from a local video file using ffmpeg."""
    if not os.path.exists(input_path):
        print(f"[ERROR] File not found: {input_path}")
        exit(1)

    # Check ffmpeg is available
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True)
    if result.returncode != 0:
        print("[ERROR] ffmpeg not found. Install it from https://ffmpeg.org/download.html and add to PATH.")
        exit(1)

    print("  Extracting audio from local file...")
    ff_result = subprocess.run([
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",                  # no video
        "-acodec", "pcm_s16le", # 16-bit PCM
        "-ar", "16000",         # 16kHz sample rate
        "-ac", "1",             # mono
        output_wav
    ], capture_output=True, text=True)

    if ff_result.returncode != 0:
        print(f"[ERROR] ffmpeg failed: {ff_result.stderr.strip()}")
        exit(1)


def _resample_wav(input_wav: str, output_wav: str) -> None:
    """Resample any WAV to 16kHz mono using ffmpeg."""
    if input_wav == output_wav:
        # In-place resample: write to temp then rename
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
        if input_wav != output_wav and os.path.exists(input_wav):
            os.remove(input_wav)
            
            
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

        # Split at sentence boundary near the limit
        if len(current_words) >= max_words and word.endswith(('.', '?', '!')):
            chunks.append(' '.join(current_words))
            current_words = []

        # Hard cap: force split if we're 20% over max_words with no boundary found
        elif len(current_words) >= int(max_words * 1.2):
            chunks.append(' '.join(current_words))
            current_words = []

    if current_words:
        chunks.append(' '.join(current_words))

    return chunks