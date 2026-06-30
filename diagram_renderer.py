"""
diagram_renderer.py
-------------------
Renders Mermaid.js diagram definitions into PNG images.
Uses the Mermaid CLI (mmdc) via subprocess.
Falls back to a placeholder if rendering fails — pipeline never stops for a bad diagram.
"""

import os
import subprocess
import tempfile
from pathlib import Path


def render_diagrams(blocks: list[dict], output_dir: str, lecture_title: str) -> dict[int, str]:
    """
    Find all diagram blocks and render them to PNG files.
    Returns a mapping of block index -> PNG file path.
    Blocks that fail to render are skipped (placeholder shown in PDF instead).
    """
    diagrams_dir = os.path.join(output_dir, "diagrams")
    os.makedirs(diagrams_dir, exist_ok=True)

    index_to_path = {}

    for i, block in enumerate(blocks):
        if block.get("type") != "diagram":
            continue

        mermaid_code = block.get("diagram_mermaid")
        if not mermaid_code:
            continue

        safe_title = _sanitize(lecture_title)[:30]
        output_png = os.path.join(diagrams_dir, f"{safe_title}_diagram_{i}.png")

        print(f"    Rendering diagram {i} ({block.get('diagram_type', 'unknown')})...")
        success = _render_mermaid(mermaid_code, output_png)

        if success:
            index_to_path[i] = output_png
        else:
            print(f"    [WARN] Diagram {i} failed to render. Placeholder will be used.")

    return index_to_path


def _render_mermaid(mermaid_code: str, output_png: str) -> bool:
    """
    Write Mermaid code to a temp file and call mmdc to render it as PNG.
    Returns True on success, False on failure.
    """
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mmd", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(mermaid_code)
            tmp_path = tmp.name
        # Use full path to mmdc on Windows (installed as .ps1 via npm)
        mmdc_cmd = r"C:\Users\vanshkumar\AppData\Roaming\npm\mmdc.ps1"

        result = subprocess.run([
            "powershell", "-ExecutionPolicy", "Bypass",
            "-File", mmdc_cmd,
            "-i", tmp_path,
            "-o", output_png,
            "-b", "white",
            "--width", "1200",
            "--height", "800"
        ], capture_output=True, text=True, timeout=60)

        os.unlink(tmp_path)

        if result.returncode != 0:
            print(f"      mmdc error: {result.stderr.strip()[:200]}")
            return False

        return os.path.exists(output_png)

    except subprocess.TimeoutExpired:
        print("      mmdc timed out.")
        return False
    except Exception as e:
        print(f"      Diagram render exception: {e}")
        return False


def _sanitize(text: str) -> str:
    """Remove special characters for use in filenames."""
    return "".join(c if c.isalnum() or c in "_- " else "" for c in text).replace(" ", "_")