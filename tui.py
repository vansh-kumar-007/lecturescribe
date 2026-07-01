"""
tui.py
------
Terminal UI for LectureScribe.
Handles source selection, folder navigation, and pipeline configuration
before handing off to the main processing pipeline.
"""

import os
import sys
import json
from pathlib import Path

import questionary
from questionary import Style
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box

from subtitle_parser import find_subtitle_for_video, can_parse

console = Console()

# ── Custom style ───────────────────────────────────────────────────────────────
TUI_STYLE = Style([
    ("qmark",        "fg:#00bfff bold"),
    ("question",     "fg:#ffffff bold"),
    ("answer",       "fg:#00ff99 bold"),
    ("pointer",      "fg:#00bfff bold"),
    ("highlighted",  "fg:#00bfff bold"),
    ("selected",     "fg:#00ff99"),
    ("separator",    "fg:#444444"),
    ("instruction",  "fg:#888888"),
])


# ── Banner ─────────────────────────────────────────────────────────────────────
def print_banner():
    console.print()
    console.print(Panel(
        Text("LectureScribe v1.0", style="bold cyan", justify="center"),
        subtitle="[dim]AI-powered lecture notes · Powered by NVIDIA Nemotron[/dim]",
        expand=False,
        border_style="cyan"
    ))
    console.print()


# ── Folder browser ─────────────────────────────────────────────────────────────
def browse_folder(start_path: str = None) -> str:
    """
    Interactive folder browser.
    Lets user navigate into subfolders until they select the target directory.
    Returns the absolute path of the selected folder.
    """
    if start_path:
        current = Path(start_path).resolve()
    else:
        # Start from common locations
        current = Path.home()

    while True:
        console.print(f"\n[dim]Current location:[/dim] [cyan]{current}[/cyan]")

        # List subdirectories
        try:
            subdirs = sorted([d for d in current.iterdir() if d.is_dir()])
        except PermissionError:
            console.print("[red]Permission denied. Going up...[/red]")
            current = current.parent
            continue

        if not subdirs:
            console.print("[yellow]No subfolders found here.[/yellow]")
            choices = ["✓ Select this folder", "⬆ Go up"]
        else:
            dir_names = [f"📁 {d.name}" for d in subdirs]
            choices = ["✓ Select this folder", "⬆ Go up"] + dir_names

        answer = questionary.select(
            "Navigate to your course/day folder:",
            choices=choices,
            style=TUI_STYLE
        ).ask()

        if answer is None:
            console.print("[red]Cancelled.[/red]")
            sys.exit(0)

        if answer == "✓ Select this folder":
            return str(current)

        elif answer == "⬆ Go up":
            current = current.parent

        else:
            # Strip the emoji prefix
            folder_name = answer[2:]  # remove "📁 "
            current = current / folder_name


# ── File preview table ─────────────────────────────────────────────────────────
def show_file_preview(folder: str, use_srt: bool):
    """Show a table of files that will be processed."""
    folder_path = Path(folder)
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}

    table = Table(
        title=f"Files to process in: {folder_path.name}",
        box=box.ROUNDED,
        border_style="cyan",
        show_lines=True
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("File", style="white")
    table.add_column("Transcript Source", style="green")

    video_files = sorted([f for f in folder_path.iterdir()
                          if f.suffix.lower() in video_exts])

    if not video_files:
        console.print("[red]No video files found in this folder.[/red]")
        sys.exit(1)

    for i, vf in enumerate(video_files, 1):
        if use_srt:
            srt = _find_srt(vf)
            if srt:
                ext = Path(srt).suffix.upper()[1:]
                source = f"[green]{ext}[/green]: {Path(srt).name}"
            else:
                source = "[yellow]Whisper (no subtitle)[/yellow]"



        else:
            source = "[cyan]Whisper GPU[/cyan]"
        table.add_row(str(i), vf.name, source)

    console.print()
    console.print(table)
    console.print()


def _find_srt(video_path: Path) -> str | None:
    """Wrapper around subtitle_parser for backward compat."""
    return find_subtitle_for_video(str(video_path))


# ── Source type selection ──────────────────────────────────────────────────────
def select_source() -> dict:
    """
    Main TUI flow. Returns a config dict:
    {
        "mode": "folder" | "url",
        "input": <path or URL>,
        "use_srt": True | False,   # only for folder mode
        "title_override": str | None
    }
    """
    print_banner()

    # Step 1: Source type
    source_type = questionary.select(
        "What is your lecture source?",
        choices=[
            "📁  Local Course Folder",
            "▶   YouTube URL",
            "🌐  Other URL (direct video link)",
        ],
        style=TUI_STYLE
    ).ask()

    if source_type is None:
        sys.exit(0)

    # ── Local folder flow ──────────────────────────────────────────────────────
    if source_type.startswith("📁"):
        # Ask for root path or browse
        root_input = questionary.text(
            "Enter course root path (or press Enter to browse from home):",
            style=TUI_STYLE
        ).ask()

        if root_input is None:
            sys.exit(0)

        start = root_input.strip() if root_input.strip() else None
        folder = browse_folder(start)

        # Transcription source
        trans_choice = questionary.select(
            "Transcription source:",
            choices=[
                "⚡  SRT Subtitles (instant, recommended if available)",
                "🎙  Video files via Whisper GPU (accurate but slow)",
            ],
            style=TUI_STYLE
        ).ask()

        if trans_choice is None:
            sys.exit(0)

        use_srt = trans_choice.startswith("⚡")

        # Preview files
        show_file_preview(folder, use_srt)

        # Confirm
        confirmed = questionary.confirm(
            "Proceed with these files?",
            default=True,
            style=TUI_STYLE
        ).ask()

        if not confirmed:
            console.print("[yellow]Aborted.[/yellow]")
            sys.exit(0)

        return {
            "mode": "folder",
            "input": folder,
            "use_srt": use_srt,
            "title_override": None
        }

    # ── URL flow ───────────────────────────────────────────────────────────────
    else:
        is_youtube = source_type.startswith("▶")
        prompt = "YouTube URL:" if is_youtube else "Video URL:"

        url = questionary.text(prompt, style=TUI_STYLE).ask()

        if not url or not url.strip():
            console.print("[red]No URL provided. Exiting.[/red]")
            sys.exit(1)

        title = questionary.text(
            "Lecture title (optional, press Enter to skip):",
            style=TUI_STYLE
        ).ask()

        return {
            "mode": "url",
            "input": url.strip(),
            "use_srt": False,
            "title_override": title.strip() if title and title.strip() else None
        }