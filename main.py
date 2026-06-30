"""
main.py
-------
LectureScribe entry point.
Launches TUI for source selection, then runs the full pipeline.
"""

import sys
import os
import time
import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from tui import select_source
from utils import extract_audio, get_folder_transcript
from transcriber import transcribe
from nemotron import analyze_transcript
from diagram_renderer import render_diagrams
from pdf_renderer import render_pdf

console = Console()


def step(num: int, total: int, emoji: str, label: str):
    console.print(f"[{num}/{total}] {emoji}  {label}...", end=" ")
    start = time.time()

    def done(note: str = ""):
        elapsed = time.time() - start
        suffix = f"({note})" if note else f"({elapsed:.1f}s)"
        console.print(f"[green]✓ Done[/green] {suffix}")

    def fail(msg: str):
        console.print(f"[red]✗ Failed[/red]")
        console.print(f"[red]  {msg}[/red]")
        sys.exit(1)

    return done, fail


def _save_transcript(transcript: str):
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)


def main():
    # ── TUI: get config ────────────────────────────────────────────────────────
    config = select_source()

    console.print()
    console.rule("[cyan]Starting Pipeline[/cyan]")
    console.print()

    # ── Step 1: Get transcript ─────────────────────────────────────────────────
    done, fail = step(1, 4, "📝", "Getting transcript")
    try:
        if config["mode"] == "folder":
            transcript, folder_title = get_folder_transcript(
                config["input"],
                use_srt=config["use_srt"]
            )
            title_hint = config["title_override"] or folder_title
        else:
            # URL mode: extract audio then transcribe
            audio_path = extract_audio(config["input"])
            transcript = transcribe(audio_path)
            title_hint = config["title_override"] or "Lecture"

        _save_transcript(transcript)
        done(f"~{len(transcript.split())} words")
    except Exception as e:
        fail(str(e))

    # ── Step 2: Nemotron analysis ──────────────────────────────────────────────
    done, fail = step(2, 4, "🧠", "Analyzing with Nemotron")
    try:
        notes = analyze_transcript(transcript)
        if config["title_override"]:
            notes["lecture_title"] = config["title_override"]
        with open("outputs/notes.json", "w", encoding="utf-8") as f:
            json.dump(notes, f, indent=2)
        done(f"{len(notes.get('blocks', []))} blocks")
    except Exception as e:
        fail(str(e))

    # ── Step 3: Diagram rendering ──────────────────────────────────────────────
    done, fail = step(3, 4, "📊", "Rendering diagrams")
    try:
        diagram_paths = render_diagrams(
            notes["blocks"], "outputs", notes.get("lecture_title", "lecture")
        )
        done(f"{len(diagram_paths)} diagram(s)")
    except Exception as e:
        fail(str(e))

    # ── Step 4: PDF generation ─────────────────────────────────────────────────
    done, fail = step(4, 4, "📄", "Generating PDF")
    try:
        pdf_path = render_pdf(notes, diagram_paths, "outputs")
        done()
    except Exception as e:
        fail(str(e))

    # ── Done ───────────────────────────────────────────────────────────────────
    console.print()
    console.print(f"[bold yellow]✨ Notes saved to:[/bold yellow] {pdf_path}")
    os.startfile(os.path.abspath(pdf_path))


if __name__ == "__main__":
    main()