"""
PDF page rendering with text highlighting.

Given a search result (PDF + page + snippet), this module:
  1. Opens the PDF at the target page
  2. Locates the snippet using a cascading match strategy
  3. Draws highlight rectangles over any matches
  4. Renders the annotated page to PNG bytes for display in Panel

All matching runs against the live PDF via PyMuPDF's ``page.search_for``,
which operates on the document's original text layout — so we do not need
the cleaned chunk text to be character-exact with what appears in the PDF.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF


# Yellow highlight, RGB in [0, 1]
_HIGHLIGHT_COLOUR = (1.0, 0.92, 0.2)

# Cap snippet length passed to search_for (longer needles rarely match and are slow)
_MAX_PROBE_CHARS = 200


@dataclass
class PreviewResult:
    """Result of a page preview render."""

    png_bytes: bytes
    match_count: int
    match_strategy: str   # "exact" | "trimmed" | "sentence" | "keyword" | "none"
    page_number: int
    doc_name: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_snippet(s: str) -> str:
    """Collapse whitespace; strip."""
    return re.sub(r"\s+", " ", s).strip()


def _find_matches(page, snippet: str) -> tuple[list, str]:
    """
    Apply matching strategies in order of decreasing precision.

    Returns
    -------
    (rects, strategy)
        ``rects`` is a list of ``fitz.Rect`` objects to highlight.
        ``strategy`` is the name of the strategy that produced them,
        or ``"none"`` if nothing matched.
    """
    snippet = _clean_snippet(snippet)
    if not snippet:
        return [], "none"

    # 1. Exact match (capped length for performance)
    probe = snippet[:_MAX_PROBE_CHARS]
    rects = page.search_for(probe)
    if rects:
        return rects, "exact"

    # 2. Trimmed 80-char prefix, backed off to the last whole word
    trim = snippet[:80]
    if " " in trim:
        trim = trim.rsplit(" ", 1)[0]
    rects = page.search_for(trim)
    if rects:
        return rects, "trimmed"

    # 3. First sentence
    sentences = re.split(r"(?<=[.!?])\s+", snippet)
    if sentences and len(sentences[0].split()) >= 4:
        rects = page.search_for(sentences[0])
        if rects:
            return rects, "sentence"

    # 4. Keyphrase fallback: first few 3-word windows
    words = snippet.split()
    combined: list = []
    for i in range(min(len(words) - 2, 10)):
        phrase = " ".join(words[i : i + 3])
        combined.extend(page.search_for(phrase))
    if combined:
        return combined, "keyword"

    return [], "none"


def _render_png(page, dpi: int) -> bytes:
    """Rasterise *page* to PNG bytes at the requested DPI."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return pix.tobytes("png")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_page_with_highlights(
    pdf_path: str | Path,
    page_number: int,
    snippet: str = "",
    dpi: int = 150,
) -> PreviewResult:
    """
    Render a single PDF page as PNG, highlighting *snippet* where found.

    Parameters
    ----------
    pdf_path : path-like
        Location of the source PDF file.
    page_number : int
        1-based page number to render.  Values outside the document range
        are clamped to the nearest valid page.
    snippet : str, optional
        Text to locate and highlight on the page.  If empty, the page is
        rendered without any annotations.
    dpi : int, default ``150``
        Render resolution.  Higher = sharper but slower and larger.

    Returns
    -------
    PreviewResult
        PNG bytes plus metadata describing which strategy matched and how
        many rectangles were highlighted.  When ``match_count == 0`` the
        page is rendered un-annotated and the UI should display a gentle
        fallback message (e.g. "Exact highlight unavailable").
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    try:
        idx = max(0, min(page_number - 1, len(doc) - 1))
        page = doc[idx]

        rects: list = []
        strategy = "none"
        if snippet:
            rects, strategy = _find_matches(page, snippet)

        for rect in rects:
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=_HIGHLIGHT_COLOUR)
            annot.update()

        png_bytes = _render_png(page, dpi)

    finally:
        doc.close()

    return PreviewResult(
        png_bytes=png_bytes,
        match_count=len(rects),
        match_strategy=strategy,
        page_number=idx + 1,
        doc_name=pdf_path.name,
    )
