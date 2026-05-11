"""Canonical preprocessing pipeline for COP509 PDF ingestion.

This module preserves the working preprocessing behaviour from the project
prototype while making it safe to import from the application, notebooks,
and backend modules.
"""

from __future__ import annotations

import io
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF

try:
    import pytesseract
    from PIL import Image
except ImportError:  # pragma: no cover - dependency availability is environment-specific
    pytesseract = None
    Image = None


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PageText:
    """Container for page-level extracted text with traceable metadata."""

    doc_id: str
    page: int
    text: str


@dataclass(frozen=True)
class PageExtractionRecord:
    """Internal page record with OCR provenance preserved."""

    doc_id: str
    page: int
    text: str
    ocr: bool


def extract_body_text_without_footnotes(page, header_ratio: float = 0.08, footer_ratio: float = 0.08) -> str:
    """
    Keep main-body spans and drop footnote/reference spans using layout:
    - remove header band
    - detect typical body font size (median)
    - drop spans that are (small font) and (low on page)
    """
    text_dict = page.get_text("dict")
    page_height = page.rect.height

    spans: list[tuple[str, float, float, float, float, float]] = []
    for block in text_dict.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if not text or not text.strip():
                    continue
                x0, y0, x1, y1 = span["bbox"]
                if y1 <= header_ratio * page_height:
                    continue
                if y0 >= (1.0 - footer_ratio) * page_height:
                    continue
                spans.append((text, x0, y0, x1, y1, float(span.get("size", 0.0))))

    if not spans:
        return ""

    sizes = sorted(span[-1] for span in spans if span[-1] > 0)
    if not sizes:
        return ""
    body_size = sizes[len(sizes) // 2]

    small_thresh = body_size * 0.90
    low_y_thresh = page_height * 0.70

    kept: list[tuple[float, float, str]] = []
    for text, x0, y0, _x1, _y1, size in spans:
        if (size > 0 and size <= small_thresh) and (y0 >= low_y_thresh):
            continue
        kept.append((y0, x0, text))

    kept.sort(key=lambda item: (item[0], item[1]))
    output = "\n".join(text.strip() for _, _, text in kept if text.strip())
    output = re.sub(r"[ \t]+", " ", output)
    output = re.sub(r"\n{3,}", "\n\n", output).strip()
    return output


def doc_id_from_path(pdf_path: str | Path) -> str:
    """Create a stable document id derived from the PDF filename."""
    name = os.path.basename(str(pdf_path))
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return name


_PAGE_TOKEN_ONLY = re.compile(r"^([0-9]{1,4}|[ivxlcdm]{1,8})$", re.IGNORECASE)
_END_PAGE_TOKEN = re.compile(r"\s([0-9]{1,4}|[ivxlcdm]{1,8})\s*$", re.IGNORECASE)


def looks_like_contents_page(text: str, page_index_zero_based: int) -> bool:
    if not text:
        return False

    normalised = re.sub(r"[ \t]+", " ", text).strip()
    lines = [line.strip() for line in normalised.splitlines() if line.strip()]
    if not lines:
        lines = [normalised]

    scan = lines[:180]

    single_line_hits = 0
    pair_hits = 0

    for line in scan:
        if 8 <= len(line) <= 180 and _END_PAGE_TOKEN.search(line):
            single_line_hits += 1

    for i in range(len(scan) - 1):
        entry = scan[i]
        next_line = scan[i + 1]
        if entry.lower() == "contents":
            continue
        if 8 <= len(entry) <= 180 and _PAGE_TOKEN_ONLY.match(next_line):
            pair_hits += 1

    total_hits = single_line_hits + pair_hits
    head = " ".join(lines[:20]).lower()
    has_contents_header = "contents" in head

    if has_contents_header and total_hits >= 6:
        return True

    if page_index_zero_based <= 10:
        chapter_like = sum(
            1
            for line in scan[:80]
            if line.lower().startswith(("chapter", "section", "part"))
        )
        if total_hits >= 10 and chapter_like >= 3:
            return True

    numbered_heads = len(re.findall(r"\b\d{1,2}\.\s+[A-Z][A-Z\s&\-]{3,}", normalised))
    alpha_heads = len(re.findall(r"\b[a-g]\.\s+[A-Z][A-Za-z\-\s]{3,}", normalised))
    alpha_dense = len(re.findall(r"\b[a-g]\.\s+", normalised))

    if has_contents_header and ((numbered_heads >= 4 and alpha_heads >= 3) or (numbered_heads >= 3 and alpha_dense >= 6)):
        return True

    appendix_hits = len(re.findall(r"\bappendix\s+\d+\s*:", normalised, re.IGNORECASE))
    figure_box_hits = len(re.findall(r"\b(?:figure|box)\s+\d+\b", normalised, re.IGNORECASE))

    if appendix_hits >= 3 and (
        "summary of conclusions and recommendations" in normalised.lower()
        or figure_box_hits >= 2
    ):
        return True

    return False


def looks_like_title_page(page_index_zero_based: int, text: str) -> bool:
    """
    Detect cover/title-style pages that often appear in the first few pages.
    Conservative: avoids removing pages such as Foreword or Introduction.
    """
    if page_index_zero_based > 2 or not text:
        return False

    lowered = text.lower()
    if any(keyword in lowered for keyword in ["foreword", "introduction", "recommendation", "executive summary"]):
        return False

    words = re.findall(r"\w+", text)
    if len(words) > 160:
        return False

    punctuation = sum(text.count(ch) for ch in ".;:?!")
    if punctuation > 4:
        return False

    title_signals = [
        "presented to parliament",
        "by command of",
        "ordered by",
        "report",
        "volume",
        "hc ",
        "cp",
    ]
    return any(signal in lowered for signal in title_signals)


def looks_like_metadata_page(text: str) -> bool:
    """Detect boilerplate licensing or copyright pages."""
    if not text:
        return False

    lowered = text.lower()
    indicators = [
        "crown copyright",
        "open government licence",
        "isbn",
        "printed in the uk",
        "nationalarchives.gov.uk",
    ]
    matches = sum(1 for phrase in indicators if phrase in lowered)
    return matches >= 2


def looks_like_serial_id_page(text: str) -> bool:
    """
    Detect near-empty identifier pages such as serial code plus ISBN.
    """
    if not text:
        return False

    raw = str(text)
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines or len(lines) > 4:
        return False

    serial_like = bool(re.search(r"\b[A-Z]\d{6,12}\b", raw, flags=re.I))
    isbn_like = bool(
        re.search(r"\b97[89][\-\s]?\d(?:[\-\s]?\d){9,12}\b", raw)
        or re.search(r"\bisbn\b", raw, re.I)
    )
    non_identifier_words = len(re.findall(r"[A-Za-z]{3,}", raw))
    return serial_like and isbn_like and non_identifier_words <= 6


def remove_repeated_lines(
    pages: list[PageExtractionRecord],
    min_doc_freq: float = 0.5,
    max_line_len: int = 140,
) -> list[PageExtractionRecord]:
    """
    Remove short-to-medium lines that repeat across many pages of the same
    document (common headers and footers that slip through positional filtering).
    """
    if not pages:
        return pages

    line_freq: dict[str, int] = {}
    for page in pages:
        unique_lines = {line.strip() for line in page.text.splitlines() if line.strip()}
        for line in unique_lines:
            if len(line) > max_line_len:
                continue
            if re.fullmatch(r"\d{1,3}", line):
                continue
            line_freq[line] = line_freq.get(line, 0) + 1

    n_pages = len(pages)
    repeated = {line for line, count in line_freq.items() if (count / n_pages) >= min_doc_freq}

    cleaned: list[PageExtractionRecord] = []
    for page in pages:
        lines = [line for line in page.text.splitlines() if line.strip()]
        kept = [line for line in lines if line.strip() not in repeated]
        new_text = "\n".join(kept).strip()
        if new_text:
            cleaned.append(
                PageExtractionRecord(
                    doc_id=page.doc_id,
                    page=page.page,
                    text=new_text,
                    ocr=page.ocr,
                )
            )
    return cleaned


def remove_cross_page_paragraph_references(pages: list[PageExtractionRecord]) -> list[PageExtractionRecord]:
    """
    Simple rule:
    - If '(paragraph' is the last thing on a page, remove that tail.
    - On following page(s), remove everything up to and including the first ')'.
    """
    if not pages:
        return pages

    cleaned: list[PageExtractionRecord] = []
    trim_until_close_bracket = False

    for page in pages:
        text = page.text or ""

        if trim_until_close_bracket:
            match_close = re.search(r"\)", text)
            if match_close:
                text = text[match_close.end() :].lstrip(" .,:;-\n\t")
                trim_until_close_bracket = False

        match_tail = re.search(r"\(\s*paragraphs?\s*$", text, flags=re.IGNORECASE)
        if match_tail:
            text = text[: match_tail.start()].rstrip(" .,:;-\n\t")
            trim_until_close_bracket = True

        cleaned.append(
            PageExtractionRecord(
                doc_id=page.doc_id,
                page=page.page,
                text=text,
                ocr=page.ocr,
            )
        )

    return cleaned


def remove_chapter_bracket_sentences(text: str) -> str:
    """
    Rules:
    - Keep spans containing 'CHAPTER' in all caps.
    - If a span contains '(Chapter ...)', remove from the start of that span
      up to the next numeric label such as 8., 8.1, 8.2., 8.1.)
    - Keep the numeric label.
    - If no later numeric label exists, remove the whole span containing
      '(Chapter ...)'.
    """
    if not text:
        return text

    parts = re.split(r"(?<=[.!?])\s+", text)
    kept: list[str] = []

    for sentence in parts:
        sentence = sentence.strip()
        if not sentence:
            continue

        if re.search(r"\bCHAPTER\b", sentence):
            kept.append(sentence)
            continue

        cleaned = re.sub(
            r"^.*?\(\s*Chapter[^)]*\).*?(?=(\d+(?:\.\d+)*\.?[)\]]?))",
            "",
            sentence,
            flags=re.IGNORECASE,
        ).strip()

        if re.search(r"\(\s*Chapter[^)]*\)", cleaned, flags=re.IGNORECASE):
            continue

        if cleaned:
            kept.append(cleaned)

    output = " ".join(kept)
    output = re.sub(r"\s{2,}", " ", output).strip()
    return output


def clean_ocr_artifacts(text: str) -> str:
    """
    Remove common OCR-only header or footer artifacts that survive page cropping.
    """
    if not text:
        return text
    output = str(text)
    output = re.sub(
        r"^\s*\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*(?:AM|PM)?\s*",
        "",
        output,
        flags=re.I,
    )
    output = re.sub(r"\bGovernment response to the .*?GOV\.UK\b", "", output, flags=re.I)
    output = re.sub(r"\b\d+\s*/\s*\d+\s*$", "", output)
    output = re.sub(r"[ \t]+", " ", output)
    output = re.sub(r"\n{3,}", "\n\n", output)
    return output.strip()


def strip_ocr_metadata_text(text: str) -> str:
    """
    For OCR pages, remove common copyright or licensing boilerplate while keeping
    the page because it may also contain valid content.
    """
    if not text:
        return text
    output = str(text)
    patterns = [
        r"all content is available under the open government licence v?3\.0[^.\n]*\.?",
        r"open government licence v?3\.0[^.\n]*\.?",
        r"\bogl\b",
        r"\u00c2?\u00a9?\s*crown copyright[^.\n]*\.?",
        r"crown copyright[^.\n]*\.?",
        r"nationalarchives\.gov\.uk[^ \n]*",
        r"https?://www\.gov\.uk/government/publications/[^\s]+",
    ]
    for pattern in patterns:
        output = re.sub(pattern, "", output, flags=re.I)
    output = re.sub(r"[ \t]+", " ", output)
    output = re.sub(r"\n{3,}", "\n\n", output)
    return output.strip()


def _run_page_ocr(page, page_height: float, ocr_header_ratio: float, ocr_footer_ratio: float) -> str:
    if pytesseract is None or Image is None:
        raise RuntimeError("OCR fallback requires pytesseract and Pillow to be installed.")

    matrix = fitz.Matrix(2, 2)
    y_top = page.rect.y0 + (ocr_header_ratio * page_height)
    y_bottom = page.rect.y1 - (ocr_footer_ratio * page_height)
    clip = fitz.Rect(page.rect.x0, y_top, page.rect.x1, y_bottom)
    if clip.y1 <= clip.y0:
        clip = page.rect
    pixmap = page.get_pixmap(matrix=matrix, alpha=False, clip=clip)
    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    return pytesseract.image_to_string(image).strip()


def _normalise_page_text(text: str) -> str:
    output = text.replace("\u00ad", "")
    output = output.replace("\x07", "")
    output = re.sub(r"\bHol\s+Inquiry\b", "", output, flags=re.I)
    output = re.sub(
        r"\bGrowing\s+the\s+UK(?:['\u2019]s)?\s+space\s+economy\b(?=\s+Recommendation\b)",
        "",
        output,
        flags=re.I,
    )
    output = re.sub(
        r"\bLeveraging\s+international\s+partnerships\b(?=\s+Recommendation\b)",
        "",
        output,
        flags=re.I,
    )
    output = re.sub(
        r"\bSecuring\s+a\s+safe\s+operating\s+environment\s+in\s+space\b(?=\s+Recommendation\b)",
        "",
        output,
        flags=re.I,
    )
    output = output.replace("\uf0b7", "\u2022")
    output = re.sub(r"(\w)-\s+(\w)", r"\1-\2", output)
    output = re.sub(r"\[\s*(?:\.|\u2026)+\s*\]", "", output)
    output = re.sub(r"[ \t]+", " ", output)
    output = re.sub(r"\n{3,}", "\n\n", output)
    output = output.strip()
    output = re.sub(r"(?<=[A-Za-z])\.\s*\d{1,3}(?![\d.])", ".", output)
    output = re.sub(r"(?m)^\s*\d{1,3}\s*$\n?", "", output)
    output = re.sub(r"(?<!\n)\n(?!\n)", " ", output)
    output = re.sub(r"\s*\([^)]*\bparagraphs?\b[^)]*\)", "", output, flags=re.IGNORECASE)
    output = re.sub(r",\s*,+", ", ", output)
    output = remove_chapter_bracket_sentences(output)
    return output


def extract_pdf_page_records(
    pdf_path: str | Path,
    doc_id: str | None = None,
    use_ocr_fallback: bool = True,
    header_ratio: float = 0.08,
    footer_ratio: float = 0.08,
    ocr_header_ratio: float = 0.03,
    ocr_footer_ratio: float = 0.03,
) -> tuple[list[PageExtractionRecord], int]:
    """
    Extract text page-by-page with the canonical prototype preprocessing logic.

    This is the import-safe backend entry point used by the application.
    """
    pdf_path = Path(pdf_path)
    resolved_doc_id = doc_id or doc_id_from_path(pdf_path)
    doc = fitz.open(str(pdf_path))
    pages: list[PageExtractionRecord] = []
    ocr_pages_used = 0
    warned_missing_ocr = False

    for i in range(len(doc)):
        page = doc[i]
        page_height = page.rect.height
        used_ocr_this_page = False

        text = extract_body_text_without_footnotes(
            page,
            header_ratio=header_ratio,
            footer_ratio=footer_ratio,
        )

        if use_ocr_fallback and not text:
            try:
                text = _run_page_ocr(page, page_height, ocr_header_ratio, ocr_footer_ratio)
            except Exception as exc:
                if not warned_missing_ocr:
                    logger.warning("%s OCR fallback unavailable: %s", pdf_path.name, exc)
                    warned_missing_ocr = True
                text = ""
            else:
                text = clean_ocr_artifacts(text)
                if text:
                    used_ocr_this_page = True
                    ocr_pages_used += 1

        text = _normalise_page_text(text)

        if looks_like_contents_page(text, i):
            continue
        if looks_like_title_page(i, text):
            continue
        if looks_like_serial_id_page(text):
            continue
        if looks_like_metadata_page(text):
            if used_ocr_this_page:
                text = strip_ocr_metadata_text(text)
            else:
                continue

        if text:
            pages.append(
                PageExtractionRecord(
                    doc_id=resolved_doc_id,
                    page=i + 1,
                    text=text,
                    ocr=used_ocr_this_page,
                )
            )

    doc.close()

    pages = remove_repeated_lines(pages, min_doc_freq=0.5)
    pages = remove_cross_page_paragraph_references(pages)
    return pages, ocr_pages_used


def extract_pdf_pages(
    pdf_path: str | Path,
    doc_id: str | None = None,
    use_ocr_fallback: bool = True,
    header_ratio: float = 0.08,
    footer_ratio: float = 0.08,
    ocr_header_ratio: float = 0.03,
    ocr_footer_ratio: float = 0.03,
) -> tuple[list[PageText], int]:
    """
    Preserve the original prototype interface: return cleaned pages and OCR count.
    """
    page_records, ocr_pages_used = extract_pdf_page_records(
        pdf_path=pdf_path,
        doc_id=doc_id,
        use_ocr_fallback=use_ocr_fallback,
        header_ratio=header_ratio,
        footer_ratio=footer_ratio,
        ocr_header_ratio=ocr_header_ratio,
        ocr_footer_ratio=ocr_footer_ratio,
    )
    pages = [PageText(doc_id=page.doc_id, page=page.page, text=page.text) for page in page_records]
    return pages, ocr_pages_used


def collect_pages_from_pdfs(pdf_paths: Iterable[str | Path]) -> tuple[list[PageText], dict[str, int]]:
    """
    Build a page collection from multiple PDFs using the canonical extractor.
    """
    all_pages: list[PageText] = []
    ocr_usage_by_doc: dict[str, int] = {}

    for pdf_path in pdf_paths:
        doc_id = doc_id_from_path(pdf_path)
        pages, ocr_pages_used = extract_pdf_pages(pdf_path, doc_id=doc_id)
        all_pages.extend(pages)
        if ocr_pages_used > 0:
            ocr_usage_by_doc[doc_id] = ocr_pages_used

    return all_pages, ocr_usage_by_doc


__all__ = [
    "PageText",
    "PageExtractionRecord",
    "collect_pages_from_pdfs",
    "clean_ocr_artifacts",
    "doc_id_from_path",
    "extract_body_text_without_footnotes",
    "extract_pdf_page_records",
    "extract_pdf_pages",
    "looks_like_contents_page",
    "looks_like_metadata_page",
    "looks_like_serial_id_page",
    "looks_like_title_page",
    "remove_chapter_bracket_sentences",
    "remove_cross_page_paragraph_references",
    "remove_repeated_lines",
    "strip_ocr_metadata_text",
]
