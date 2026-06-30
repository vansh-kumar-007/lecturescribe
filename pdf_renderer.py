"""
pdf_renderer.py
---------------
Renders structured notes JSON into a printable A4 PDF using ReportLab.
Styled to look like handwritten notes: color-coded by block type,
with Patrick Hand font (Google Fonts), diagrams embedded inline.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor, grey, white, black
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, HRFlowable,
    PageBreak, Table, TableStyle
)
from reportlab.platypus.flowables import Flowable
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


# ── Colors ────────────────────────────────────────────────────────────────────
C_HEADING    = HexColor("#1e3a6e")
C_BODY       = HexColor("#1a4fa0")
C_IMPORTANT  = HexColor("#c0392b")
C_CRITICAL   = HexColor("#6c3483")
C_CRITICAL_BG= HexColor("#f5eef8")
C_NOTE       = HexColor("#2c7a2c")
C_GRAY       = HexColor("#888888")
C_RULE       = HexColor("#cccccc")


# ── Font registration ──────────────────────────────────────────────────────────
def _register_fonts():
    """
    Attempt to register Patrick Hand from local cache.
    Falls back to Helvetica if unavailable (always works in ReportLab).
    """
    try:
        font_path = _download_font_if_needed()
        if font_path:
            pdfmetrics.registerFont(TTFont("PatrickHand", font_path))
            return "PatrickHand"
    except Exception:
        pass
    return "Helvetica"


def _download_font_if_needed() -> str | None:
    """Download Patrick Hand font from Google Fonts if not cached."""
    import urllib.request
    cache_dir = Path("outputs/fonts")
    cache_dir.mkdir(exist_ok=True)
    font_file = cache_dir / "PatrickHand-Regular.ttf"

    if font_file.exists():
        return str(font_file)

    url = "https://github.com/google/fonts/raw/main/ofl/patrickhand/PatrickHand-Regular.ttf"
    try:
        print("  Downloading Patrick Hand font...")
        urllib.request.urlretrieve(url, font_file)
        return str(font_file)
    except Exception as e:
        print(f"  [WARN] Could not download font: {e}. Using Helvetica.")
        return None


# ── Page numbering ─────────────────────────────────────────────────────────────
class _PageNumCanvas(canvas.Canvas):
    """Canvas subclass that adds 'Page X of Y' footer to every page."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pages = []

    def showPage(self):
        self._pages.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        total = len(self._pages)
        for i, page in enumerate(self._pages):
            self.__dict__.update(page)
            self._draw_page_number(i + 1, total)
            super().showPage()
        super().save()

    def _draw_page_number(self, page_num: int, total: int):
        self.setFont("Helvetica", 9)
        self.setFillColor(C_GRAY)
        text = f"Page {page_num} of {total}"
        self.drawCentredString(A4[0] / 2, 10 * mm, text)


# ── Critical block (boxed) ─────────────────────────────────────────────────────
class _CriticalBox(Flowable):
    """A purple-bordered box for critical blocks."""

    def __init__(self, text: str, font: str, width: float):
        super().__init__()
        self._text = text
        self._font = font
        self._width = width
        self.hAlign = "LEFT"

    def wrap(self, availWidth, availHeight):
        self._w = min(self._width, availWidth)
        self._h = 40
        return self._w, self._h

    def draw(self):
        c = self.canv
        padding = 8
        c.setFillColor(C_CRITICAL_BG)
        c.setStrokeColor(C_CRITICAL)
        c.setLineWidth(2)
        c.roundRect(0, 0, self._w, self._h, 6, fill=1, stroke=1)
        c.setFillColor(C_CRITICAL)
        c.setFont(self._font, 11)
        c.drawString(padding, self._h / 2 - 5, self._text[:120])


# ── Style builder ──────────────────────────────────────────────────────────────
def _build_styles(font: str) -> dict:
    base = dict(fontName=font, leading=18, spaceAfter=4)
    return {
        "heading": ParagraphStyle("heading",
            fontName=font, fontSize=18, textColor=C_HEADING,
            underline=True, spaceBefore=18, spaceAfter=6, leading=22),
        "subheading": ParagraphStyle("subheading",
            fontName=font, fontSize=14, textColor=C_HEADING,
            spaceBefore=10, spaceAfter=4, leading=18),
        "body": ParagraphStyle("body",
            fontName=font, fontSize=11, textColor=C_BODY,
            spaceAfter=4, leading=18, alignment=TA_JUSTIFY),
        "important": ParagraphStyle("important",
            fontName=font, fontSize=11, textColor=C_IMPORTANT,
            spaceAfter=4, leading=18),
        "note": ParagraphStyle("note",
            fontName=font, fontSize=10, textColor=C_NOTE,
            spaceAfter=4, leading=16, leftIndent=20, fontStyle="italic"),
        "caption": ParagraphStyle("caption",
            fontName=font, fontSize=9, textColor=C_GRAY,
            spaceAfter=6, leading=12, alignment=TA_CENTER),
        "cover_title": ParagraphStyle("cover_title",
            fontName=font, fontSize=28, textColor=black,
            alignment=TA_CENTER, spaceAfter=12, leading=34),
        "cover_subject": ParagraphStyle("cover_subject",
            fontName=font, fontSize=16, textColor=C_GRAY,
            alignment=TA_CENTER, spaceAfter=8),
        "cover_date": ParagraphStyle("cover_date",
            fontName=font, fontSize=11, textColor=C_GRAY,
            alignment=TA_CENTER, spaceAfter=6),
        "cover_footer": ParagraphStyle("cover_footer",
            fontName=font, fontSize=9, textColor=C_GRAY,
            alignment=TA_CENTER),
    }


# ── Cover page ─────────────────────────────────────────────────────────────────
def _build_cover(notes: dict, styles: dict) -> list:
    date_str = datetime.now().strftime("%d %B %Y")
    return [
        Spacer(1, 80 * mm),
        Paragraph(notes.get("lecture_title", "Lecture Notes"), styles["cover_title"]),
        Paragraph(notes.get("subject", ""), styles["cover_subject"]),
        HRFlowable(width="80%", thickness=1, color=C_RULE, spaceAfter=12),
        Paragraph(f"Notes generated on {date_str}", styles["cover_date"]),
        Spacer(1, 120 * mm),
        Paragraph("Generated by LectureScribe · Powered by NVIDIA Nemotron-Ultra", styles["cover_footer"]),
        PageBreak(),
    ]


# ── Block renderer ─────────────────────────────────────────────────────────────
def _build_content(blocks: list, diagram_paths: dict, styles: dict, font: str, page_width: float) -> list:
    story = []

    for i, block in enumerate(blocks):
        btype = block.get("type", "body")
        content = block.get("content", "").strip()

        if btype == "heading":
            story.append(Spacer(1, 4))
            story.append(Paragraph(content, styles["heading"]))

        elif btype == "subheading":
            story.append(Paragraph(content, styles["subheading"]))

        elif btype == "body":
            story.append(Paragraph(content, styles["body"]))

        elif btype == "important":
            story.append(Paragraph(f"★ {content}", styles["important"]))

        elif btype == "critical":
            story.append(Spacer(1, 6))
            story.append(_CriticalBox(content, font, page_width))
            story.append(Spacer(1, 6))

        elif btype == "note":
            story.append(Paragraph(f"✎ {content}", styles["note"]))

        elif btype == "diagram":
            png_path = diagram_paths.get(i)
            description = block.get("diagram_description", "Diagram")

            if png_path and os.path.exists(png_path):
                max_w = page_width * 0.9
                img = Image(png_path, width=max_w, height=max_w * 0.6)
                img.hAlign = "CENTER"
                story.append(Spacer(1, 6))
                story.append(img)
                story.append(Paragraph(description, styles["caption"]))
                story.append(Spacer(1, 6))
            else:
                # Placeholder box
                placeholder = Paragraph(
                    f'<font color="#aaaaaa">[Diagram: {description}]</font>',
                    styles["caption"]
                )
                story.append(Spacer(1, 4))
                story.append(placeholder)
                story.append(Spacer(1, 4))

    return story


# ── Main entry point ───────────────────────────────────────────────────────────
def render_pdf(notes: dict, diagram_paths: dict, output_dir: str = "outputs") -> str:
    """
    Render the full PDF from structured notes and diagram PNGs.
    Returns the path to the saved PDF file.
    """
    os.makedirs(output_dir, exist_ok=True)

    font = _register_fonts()
    styles = _build_styles(font)

    # Output filename
    safe_title = re.sub(r"[^\w\s-]", "", notes.get("lecture_title", "Notes"))
    safe_title = re.sub(r"\s+", "_", safe_title.strip())[:50]
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_path = os.path.join(output_dir, f"{safe_title}_{date_str}.pdf")

    # Page dimensions with margins
    left_margin   = 25 * mm
    right_margin  = 18 * mm
    top_margin    = 20 * mm
    bottom_margin = 20 * mm
    page_width = A4[0] - left_margin - right_margin

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=left_margin,
        rightMargin=right_margin,
        topMargin=top_margin,
        bottomMargin=bottom_margin + 10 * mm,  # room for page numbers
    )

    story = []
    story += _build_cover(notes, styles)
    story += _build_content(notes.get("blocks", []), diagram_paths, styles, font, page_width)

    doc.build(story, canvasmaker=_PageNumCanvas)
    print(f"  PDF saved to: {output_path}")
    return output_path