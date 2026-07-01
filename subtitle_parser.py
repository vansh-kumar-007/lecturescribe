"""
subtitle_parser.py
------------------
Unified subtitle/transcript parser for LectureScribe.

Supports:
    .srt  — SubRip (most common, Udemy, VLC exports)
    .vtt  — WebVTT (YouTube auto-captions, browser downloads)
    .ass  — Advanced SubStation Alpha (anime, high-quality fansubs)
    .ssa  — SubStation Alpha (older variant of ASS)
    .txt  — Plain text transcript (no parsing needed)

Usage:
    from subtitle_parser import load_subtitle, can_parse

    text = load_subtitle("lecture.srt")   # returns clean plain text
    text = load_subtitle("captions.vtt")
    text = load_subtitle("notes.txt")
"""

import re
from pathlib import Path


# ── Public API ─────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".srt", ".vtt", ".ass", ".ssa", ".txt"}


def can_parse(file_path: str) -> bool:
    """Return True if the file extension is a supported subtitle format."""
    return Path(file_path).suffix.lower() in SUPPORTED_EXTENSIONS


def load_subtitle(file_path: str) -> str:
    """
    Parse a subtitle file and return clean plain text transcript.
    Automatically detects format from file extension.
    Raises ValueError if the format is unsupported.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {file_path}")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    if ext == ".srt":
        return _parse_srt(content)
    elif ext == ".vtt":
        return _parse_vtt(content)
    elif ext in (".ass", ".ssa"):
        return _parse_ass(content)
    elif ext == ".txt":
        return _parse_txt(content)
    else:
        raise ValueError(
            f"Unsupported subtitle format: '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def find_subtitle_for_video(video_path: str) -> str | None:
    """
    Given a video file path, search for any supported subtitle file
    with the same base name. Checks formats in priority order.
    Returns the subtitle path if found, else None.
    """
    base = str(Path(video_path).with_suffix(""))
    priority = [
        # Udemy _en suffix
        base + "_en.srt",
        base + "_en.vtt",
        # Udemy/other " - English" suffix
        base + " - English.vtt",
        base + " - English.srt",
        # Plain matches
        base + ".srt",
        base + ".vtt",
        base + ".ass",
        base + ".ssa",
        base + ".txt",
    ]
    for candidate in priority:
        if Path(candidate).exists():
            return candidate
    return None


# ── SRT parser ─────────────────────────────────────────────────────────────────

def _parse_srt(content: str) -> str:
    """
    Parse SubRip (.srt) format.

    Format:
        1
        00:00:01,000 --> 00:00:04,000
        This is the subtitle text.

        2
        00:00:05,000 --> 00:00:08,000
        Another line here.
    """
    # Remove sequence numbers (lines containing only digits)
    content = re.sub(r"^\d+\s*$", "", content, flags=re.MULTILINE)

    # Remove timestamp lines (handles both , and . as millisecond separator)
    content = re.sub(
        r"\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}.*",
        "", content
    )

    return _clean(content)


# ── VTT parser ─────────────────────────────────────────────────────────────────

def _parse_vtt(content: str) -> str:
    """
    Parse WebVTT (.vtt) format.

    Format:
        WEBVTT

        00:00:01.000 --> 00:00:04.000
        This is the subtitle text.

        NOTE Some comment

        00:00:05.000 --> 00:00:08.000
        Another line here.
    """
    # Remove WEBVTT header
    content = re.sub(r"^WEBVTT.*$", "", content, flags=re.MULTILINE)

    # Remove NOTE blocks
    content = re.sub(r"^NOTE.*$", "", content, flags=re.MULTILINE)

    # Remove STYLE blocks
    content = re.sub(r"STYLE\s*\{[^}]*\}", "", content, flags=re.DOTALL)

    # Remove cue identifiers (optional IDs before timestamps)
    content = re.sub(r"^\S+\s*$", "", content, flags=re.MULTILINE)

    # Remove timestamp lines (VTT uses . not , for milliseconds)
    content = re.sub(
        r"\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}.*",
        "", content
    )

    # Also handle short timestamp format (mm:ss.mmm)
    content = re.sub(
        r"\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}\.\d{3}.*",
        "", content
    )

    # Remove VTT cue settings tags like <c>, <b>, <i>, position tags
    content = re.sub(r"<[^>]+>", "", content)

    # Remove &nbsp; and other HTML entities
    content = re.sub(r"&\w+;", " ", content)

    return _clean(content)


# ── ASS/SSA parser ─────────────────────────────────────────────────────────────

def _parse_ass(content: str) -> str:
    """
    Parse Advanced SubStation Alpha (.ass/.ssa) format.

    ASS files have a [Events] section with Dialogue lines:
        Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,This is the text.

    SSA is similar but with slightly different field counts.
    """
    lines = []

    in_events = False
    for line in content.splitlines():
        stripped = line.strip()

        if stripped.lower() == "[events]":
            in_events = True
            continue

        if stripped.startswith("[") and in_events:
            # Entered a new section — stop
            break

        if in_events and stripped.lower().startswith("dialogue:"):
            # Extract the text field (last comma-separated field)
            # ASS format: Dialogue: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
            parts = stripped.split(",", 9)
            if len(parts) >= 10:
                text = parts[9]
                # Remove ASS override tags like {\an8}, {\pos(...)}, {\b1}, etc.
                text = re.sub(r"\{[^}]*\}", "", text)
                # Remove \N and \n (ASS line breaks)
                text = re.sub(r"\\[Nn]", " ", text)
                text = text.strip()
                if text:
                    lines.append(text)

    return " ".join(lines)


# ── TXT parser ─────────────────────────────────────────────────────────────────

def _parse_txt(content: str) -> str:
    """
    Parse plain text transcript.
    Just cleans whitespace — no timestamp removal needed.
    """
    return _clean(content)


# ── Shared cleaner ─────────────────────────────────────────────────────────────

def _clean(content: str) -> str:
    """
    Shared post-processing:
    - Remove HTML/XML tags
    - Collapse blank lines and extra whitespace
    - Join into a single space-separated string
    """
    # Remove any remaining HTML tags
    content = re.sub(r"<[^>]+>", "", content)

    # Normalize whitespace within lines
    lines = []
    for line in content.splitlines():
        line = line.strip()
        if line:
            lines.append(line)

    return " ".join(lines)
