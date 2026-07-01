"""
main.py
-------
LectureScribe entry point.
Launches TUI for source selection, creates a Job workspace,
then runs the full pipeline writing all outputs to the job folder.
"""

import sys
import os
import time
import json
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
from workspace_manager import create_job
from transcriber import transcribe, transcribe_segmented

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


def main():
    # ── TUI: get config ────────────────────────────────────────────────────────
    config = select_source()

    # ── Create job workspace ───────────────────────────────────────────────────
    title_hint = config.get("title_override") or Path(config["input"]).name
    job = create_job(
        source=config["input"],
        source_type=config["mode"],
        title=title_hint,
        config=config
    )

    logger = job.get_logger()
    logger.info(f"Job created: {job.dir}")

    console.print()
    console.rule("[cyan]Starting Pipeline[/cyan]")
    console.print(f"[dim]Workspace: {job.dir}[/dim]")
    console.print()

    # ── Step 1: Get transcript ─────────────────────────────────────────────────
    done, fail = step(1, 4, "📝", "Getting transcript")
    job.mark_step("transcription")
    try:
        if config["mode"] == "folder":
            transcript, folder_title = get_folder_transcript(
                config["input"],
                use_srt=config["use_srt"],
                job=job
            )
            if not config.get("title_override"):
                title_hint = folder_title
        else:
            audio_path = extract_audio(config["input"], job=job)
            transcript = transcribe_segmented(audio_path, job=job)
            # Save merged transcript
            with open(job.merged_transcript, "w", encoding="utf-8") as f:
                f.write(transcript)

        done(f"~{len(transcript.split())} words")
        logger.info(f"Transcript ready: {len(transcript.split())} words")
    except Exception as e:
        job.mark_failed(str(e))
        logger.error(f"Transcription failed: {e}")
        fail(str(e))

    # ── Step 2: Nemotron analysis ──────────────────────────────────────────────
    done, fail = step(2, 4, "🧠", "Analyzing with Nemotron")
    job.mark_step("nemotron")
    try:
        notes = analyze_transcript(transcript, job=job)
        if config.get("title_override"):
            notes["lecture_title"] = config["title_override"]
        done(f"{len(notes.get('blocks', []))} blocks")
        logger.info(f"Nemotron done: {len(notes.get('blocks', []))} blocks")
    except Exception as e:
        job.mark_failed(str(e))
        logger.error(f"Nemotron failed: {e}")
        fail(str(e))

    # ── Step 3: Diagram rendering ──────────────────────────────────────────────
    done, fail = step(3, 4, "📊", "Rendering diagrams")
    job.mark_step("diagrams")
    try:
        diagram_paths = render_diagrams(
            notes["blocks"],
            job=job,
            lecture_title=notes.get("lecture_title", "lecture")
        )
        done(f"{len(diagram_paths)} diagram(s)")
        logger.info(f"Diagrams rendered: {len(diagram_paths)}")
    except Exception as e:
        job.mark_failed(str(e))
        logger.error(f"Diagram rendering failed: {e}")
        fail(str(e))

    # ── Step 4: PDF generation ─────────────────────────────────────────────────
    done, fail = step(4, 4, "📄", "Generating PDF")
    job.mark_step("pdf")
    try:
        pdf_path = render_pdf(notes, diagram_paths, job=job)
        done()
        logger.info(f"PDF saved: {pdf_path}")
    except Exception as e:
        job.mark_failed(str(e))
        logger.error(f"PDF generation failed: {e}")
        fail(str(e))

    # ── Done ───────────────────────────────────────────────────────────────────
    job.mark_done()
    console.print()
    console.print(f"[bold yellow]✨ Notes saved to:[/bold yellow] {pdf_path}")
    console.print(f"[dim]Full workspace: {job.dir}[/dim]")
    os.startfile(os.path.abspath(pdf_path))


if __name__ == "__main__":
    main()
