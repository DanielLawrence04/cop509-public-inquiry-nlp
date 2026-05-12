"""
PDF ingestion wrappers around the canonical preprocessing pipeline.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TypedDict

from .preprocessing import doc_id_from_path, extract_pdf_page_records

logger = logging.getLogger(__name__)


class PageRecord(TypedDict):
    page_number: int
    text: str
    raw_text: str
    source: str
    ocr: bool


def load_pdf_text(path: str | Path, use_ocr_fallback: bool = True) -> str:
    """
    Extract the full preprocessed text of a PDF as a single string.
    """
    pages = extract_pages(path, use_ocr_fallback=use_ocr_fallback)
    return "\n".join(page["text"] for page in pages)


def extract_pages(path: str | Path, use_ocr_fallback: bool = True) -> list[PageRecord]:
    """
    Extract preprocessed page records using the canonical preprocessing module.

    The returned shape stays compatible with the rest of the coursework app.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    page_records, ocr_pages_used = extract_pdf_page_records(
        path,
        doc_id=doc_id_from_path(path),
        use_ocr_fallback=use_ocr_fallback,
    )
    records: list[PageRecord] = [
        PageRecord(
            page_number=page.page,
            text=page.text,
            raw_text=page.text,
            source=path.name,
            ocr=page.ocr,
        )
        for page in page_records
    ]

    summary = f"Loaded '{path.name}' - {len(records)} retained pages"
    if ocr_pages_used:
        summary += f", OCR applied to {ocr_pages_used} image page(s)"
    logger.info(summary)
    return records
