"""
main.py
-------
LectureScribe entry point.
Takes a local MP4 path or YouTube URL as a command-line argument
and runs the full pipeline: audio → transcription → Nemotron → diagrams → PDF.
"""

import sys
import os
import time
import json
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from utils import extract_audio, chunk_transcript
from transcriber import transcribe
from nemotron import analyze_transcript
from diagram_renderer import render_diagrams
from pdf_renderer import render_pdf

console = Console()


def print_banner():
    banner = Text("LectureScribe v1.0", style="bold cyan", justify="center")
    console.print(Panel(banner, expand=False))
    console.print()


def step(num: int, total: int, emoji: str, label: str):
    """Print a step start line and return a function to mark it done."""
    console.print(f"[{num}/{total}] {emoji}  {label}...", end=" ")
    start = time.time()

    def done(note: str = ""):
        elapsed = time.time() - start
        suffix = f"({note})" if note else f"({elapsed:.1f}s)"
        console.print(f"[green]✓ Done[/green] {suffix}")

    def fail(msg: str):
        console.print(f"[red]✗ Failed[/red]")
        console.print(f"[red]  {msg}[/red]")
        _write_error_log(msg)
        sys.exit(1)

    return done, fail


def _write_error_log(msg: str):
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/error.log", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


def _load_cached_transcript() -> str | None:
    """Return cached transcript if it exists (saves re-transcribing during dev)."""
    path = "outputs/transcript.txt"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def _save_transcript(transcript: str):
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)


def main():
    if len(sys.argv) < 2:
        console.print("[red]Usage: python main.py <video_path_or_youtube_url>[/red]")
        sys.exit(1)

    input_path = sys.argv[1]
    print_banner()

    # ── Step 1: Audio extraction ───────────────────────────────────────────────
    done, fail = step(1, 5, "🎵", "Extracting audio")
    try:
        audio_path = extract_audio(input_path)
        done()
    except SystemExit:
        fail("Audio extraction failed. Check error above.")
    except Exception as e:
        fail(str(e))

    # ── Step 2: Transcription ──────────────────────────────────────────────────
    done, fail = step(2, 5, "📝", "Transcribing with Whisper")
    try:
        transcript = transcribe(audio_path)
        _save_transcript(transcript)
        word_count = len(transcript.split())
        done(f"~{word_count} words")
    except Exception as e:
        fail(str(e))

    # ── Step 3: Nemotron analysis ──────────────────────────────────────────────
    done, fail = step(3, 5, "🧠", "Analyzing with Nemotron")
    try:
        notes = analyze_transcript(transcript)
        # Save for debugging
        with open("outputs/notes.json", "w", encoding="utf-8") as f:
            json.dump(notes, f, indent=2)
        block_count = len(notes.get("blocks", []))
        done(f"{block_count} blocks")
    except Exception as e:
        fail(str(e))

    # ── Step 4: Diagram rendering ──────────────────────────────────────────────
    done, fail = step(4, 5, "📊", "Rendering diagrams")
    try:
        diagram_paths = render_diagrams(
            notes["blocks"], "outputs", notes.get("lecture_title", "lecture")
        )
        done(f"{len(diagram_paths)} diagram(s)")
    except Exception as e:
        fail(str(e))

    # ── Step 5: PDF generation ─────────────────────────────────────────────────
    done, fail = step(5, 5, "📄", "Generating PDF")
    try:
        pdf_path = render_pdf(notes, diagram_paths, "outputs")
        done()
    except Exception as e:
        fail(str(e))

    # ── Done ───────────────────────────────────────────────────────────────────
    console.print()
    console.print(f"[bold yellow]✨ Notes saved to:[/bold yellow] {pdf_path}")

    # Open PDF automatically on Windows
    os.startfile(os.path.abspath(pdf_path))


if __name__ == "__main__":
    main()