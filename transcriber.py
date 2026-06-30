"""
transcriber.py
--------------
Handles speech-to-text transcription using OpenAI Whisper large-v3.
Runs locally on GPU (CUDA). Falls back to 'medium' model on VRAM OOM.
"""

import warnings
import whisper
import torch


def transcribe(audio_path: str) -> str:
    """
    Transcribe a WAV file using Whisper large-v3 on GPU.
    Returns the full transcript as a plain string.
    Falls back to 'medium' model if CUDA runs out of memory.
    """
    model_name = "large-v3"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"  Loading Whisper {model_name} on {device.upper()}...")

    try:
        model = _load_model(model_name, device)
        return _run_transcription(model, audio_path, model_name, device)

    except torch.cuda.OutOfMemoryError:
        print(f"  [WARN] VRAM insufficient for {model_name}. Retrying with 'medium' model...")
        torch.cuda.empty_cache()
        model_name = "medium"
        model = _load_model(model_name, device)
        return _run_transcription(model, audio_path, model_name, device)


def _load_model(model_name: str, device: str):
    """Load a Whisper model onto the specified device."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return whisper.load_model(model_name, device=device)


def _run_transcription(model, audio_path: str, model_name: str, device: str) -> str:
    """Run Whisper transcription and return plain text."""
    print(f"  Transcribing with Whisper {model_name}... (this may take a while)")

    result = model.transcribe(
        audio_path,
        fp16=(device == "cuda"),  # use FP16 on GPU for speed
        verbose=False
    )

    text = result["text"].strip()
    word_count = len(text.split())
    print(f"  Transcription complete. ~{word_count} words.")
    return text