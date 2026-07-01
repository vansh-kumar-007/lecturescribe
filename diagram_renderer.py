"""
diagram_renderer.py
-------------------
Renders Mermaid.js diagram definitions into PNG images.
Uses the Mermaid CLI (mmdc) via subprocess.
If a Job is provided, saves PNGs to job.diagram_dir and skips already-rendered ones.
Falls back to a placeholder if rendering fails — pipeline never stops for a bad diagram.
"""

import os
import subprocess
import tempfile
from pathlib import Path


def _sanitize_mermaid(code: str) -> str:
    """
    Clean up Mermaid code to prevent parse errors:
    - Wrap node labels containing special chars in double quotes
    - Remove parentheses inside node labels
    """
    import re

    def clean_label(m):
        label = m.group(1)
        label = re.sub(r'\([^)]*\)', '', label)
        label = re.sub(r'[`"\':]', '', label)
        label = re.sub(r'\s+', ' ', label).strip()
        return f'["{label}"]'

    code = re.sub(r'\[([^\]]+)\]', clean_label, code)
    return code


def render_diagrams(blocks: list[dict], output_dir: str = None, lecture_title: str = "", job=None) -> dict[int, str]:
    """
    Find all diagram blocks and render them to PNG files.
    If a Job is provided, saves to job.diagram_dir and skips already-rendered PNGs.
    Returns a mapping of block index -> PNG file path.
    """
    if job is not None:
        diagrams_dir = job.diagram_dir
    else:
        diagrams_dir = Path(output_dir) / "diagrams"
        diagrams_dir.mkdir(parents=True, exist_ok=True)

    index_to_path = {}

    for i, block in enumerate(blocks):
        if block.get("type") != "diagram":
            continue

        mermaid_code = block.get("diagram_mermaid")
        if not mermaid_code:
            continue

        if job:
            output_png = job.diagram_path(i, block.get("diagram_type", ""))
        else:
            safe_title = _sanitize(lecture_title)[:30]
            output_png = diagrams_dir / f"{safe_title}_diagram_{i}.png"

        # Resume: skip already-rendered diagrams
        if output_png.exists():
            print(f"    Diagram {i} already rendered. Skipping.")
            index_to_path[i] = str(output_png)
            continue

        print(f"    Rendering diagram {i} ({block.get('diagram_type', 'unknown')})...")
        success = _render_mermaid(mermaid_code, str(output_png))

        if success:
            index_to_path[i] = str(output_png)
        else:
            print(f"    [WARN] Diagram {i} failed to render. Placeholder will be used.")

    return index_to_path


def _render_mermaid(mermaid_code: str, output_png: str) -> bool:
    """
    Write Mermaid code to a temp file and call mmdc to render it as PNG.
    Returns True on success, False on failure.
    """
    mermaid_code = _sanitize_mermaid(mermaid_code)

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mmd", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(mermaid_code)
            tmp_path = tmp.name

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
