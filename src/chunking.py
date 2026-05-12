"""
Text chunking with configurable size and overlap.

Core chunkers
-------------
``chunk_text`` / ``chunk_pages`` — configurable sliding-window chunking.

Improved Task 1 additions
-------------------------
``chunk_pages_v2`` — improved chunker used for the final Task 1 index:
  * Default 400-word windows for richer per-chunk context.
  * ``_merge_short_chunks`` merges any chunk under *min_chunk_words* (default 60)
    with its neighbour within the same page — fixes ``chunk_too_small`` failures.
  * Page provenance and OCR flags are preserved throughout.

``detect_chunk_heading`` — lightweight heuristic that extracts a leading
section heading from a chunk's text for result display.
"""

from __future__ import annotations

import re
from typing import TypedDict

from .pdf_loader import PageRecord
from .utils import clean_text


class Chunk(TypedDict):
    chunk_id: int
    text: str
    source: str
    page_number: int | None
    ocr: bool


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    source: str = "unknown",
) -> list[Chunk]:
    """
    Split *text* into overlapping word-level chunks.

    Parameters
    ----------
    text : str
        Input text to split.
    chunk_size : int
        Target number of words per chunk.
    overlap : int
        Number of words to repeat between consecutive chunks.
    source : str
        Label attached to every chunk (e.g. filename).

    Returns
    -------
    list[Chunk]
        Ordered list of chunk dicts.
    """
    words = clean_text(text).split()
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks: list[Chunk] = []
    for i, start in enumerate(range(0, len(words), step)):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(
            Chunk(
                chunk_id=i,
                text=" ".join(chunk_words),
                source=source,
                page_number=None,
                ocr=False,
            )
        )
    return chunks


def chunk_pages(
    pages: list[PageRecord],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """
    Chunk a list of ``PageRecord`` objects, preserving page provenance.

    Parameters
    ----------
    pages : list[PageRecord]
        Output of :func:`pdf_loader.extract_pages`.
    chunk_size : int
        Target words per chunk.
    overlap : int
        Overlap words between chunks.

    Returns
    -------
    list[Chunk]
        All chunks across all pages, with ``page_number`` populated.
    """
    all_chunks: list[Chunk] = []
    global_id = 0
    for page in pages:
        page_chunks = chunk_text(
            page["text"],
            chunk_size=chunk_size,
            overlap=overlap,
            source=page["source"],
        )
        for chunk in page_chunks:
            chunk["chunk_id"] = global_id
            chunk["page_number"] = page["page_number"]
            chunk["ocr"] = page["ocr"]
            all_chunks.append(chunk)
            global_id += 1
    return all_chunks


# ===========================================================================
# Improved Task 1 chunking
# ===========================================================================

# ---------------------------------------------------------------------------
# Heading detection (display utility — does not affect retrieval)
# ---------------------------------------------------------------------------

# Patterns common in UK parliamentary / inquiry PDF reports
_HEADING_RE = re.compile(
    r'^(?:'
    r'[Rr]ecommendations?\s+\d+[:.]?'     # Recommendation 6:
    r'|[Rr]ec\s+\d+'                       # Rec 6
    r'|\d+[.\)]\s+[A-Z][A-Za-z\s]{3,}'    # 132. The Space Economy
    r'|[A-Z][A-Z\s\-,]{5,}[A-Z]'          # ALL CAPS HEADINGS
    r')',
)


def detect_chunk_heading(text: str) -> str | None:
    """
    Return a leading section heading from *text*, or ``None`` if not found.

    The heading must be in the first 120 characters and match at least one of:
    - ``Recommendation N:`` / ``Rec N``
    - A numbered heading (``132. Title``)
    - An all-caps line of 6+ characters

    This is a display-only utility used by the search widget to annotate
    result cards.  It does not alter chunk content or retrieval ranking.
    """
    first_part = text[:120].strip()
    # Try matching just the first "sentence" (up to first period or newline)
    lead = re.split(r'[.\n]', first_part, maxsplit=1)[0].strip()
    return lead if _HEADING_RE.match(lead) else None


# ---------------------------------------------------------------------------
# Short-chunk merging
# ---------------------------------------------------------------------------

def _merge_short_chunks(chunks: list[Chunk], min_words: int = 60) -> list[Chunk]:
    """
    Merge consecutive chunks where either chunk has fewer than *min_words* words,
    within the same source document and page.

    Strategy (single left-to-right pass):
    - If chunk *i* is short AND chunk *i+1* is on the same page/source,
      merge them into one chunk and advance by 2.
    - If chunk *i* is the last remaining short chunk and a previous chunk exists
      on the same page, merge it backward into the previous result.
    - Otherwise keep the chunk as-is (even if short — avoids infinite loops on
      pages with only one very short line).

    The ``page_number``, ``source``, and ``ocr`` flag from the first chunk in each
    merged pair are preserved.  ``chunk_id`` values are reassigned by the caller.
    """
    if not chunks:
        return []

    result: list[Chunk] = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i]
        n_words = len(chunk["text"].split())

        next_same_page = (
            i + 1 < len(chunks)
            and chunks[i + 1]["source"] == chunk["source"]
            and chunks[i + 1]["page_number"] == chunk["page_number"]
        )
        prev_same_page = (
            result
            and result[-1]["source"] == chunk["source"]
            and result[-1]["page_number"] == chunk["page_number"]
        )

        if n_words < min_words and next_same_page:
            # Merge forward: absorb the next chunk
            nxt = chunks[i + 1]
            merged: Chunk = {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"] + " " + nxt["text"],
                "source": chunk["source"],
                "page_number": chunk["page_number"],
                "ocr": chunk["ocr"] or nxt["ocr"],
            }
            result.append(merged)
            i += 2
        elif n_words < min_words and prev_same_page:
            # Merge backward: absorb into the previous result
            prev = result[-1]
            result[-1] = {
                "chunk_id": prev["chunk_id"],
                "text": prev["text"] + " " + chunk["text"],
                "source": prev["source"],
                "page_number": prev["page_number"],
                "ocr": prev["ocr"] or chunk["ocr"],
            }
            i += 1
        else:
            result.append(chunk)
            i += 1

    return result


# ---------------------------------------------------------------------------
# Improved page chunker
# ---------------------------------------------------------------------------

def chunk_pages_v2(
    pages: list[PageRecord],
    chunk_size: int = 400,
    overlap: int = 50,
    min_chunk_words: int = 60,
) -> list[Chunk]:
    """
    Improved chunker for the final Task 1 search index.

    Differences from :func:`chunk_pages`
    -------------------------------------
    * Default ``chunk_size=400`` provides richer per-window context,
      improving keyword coverage for paraphrase and policy-topic queries.
    * Short chunks (< *min_chunk_words* words) are merged with their nearest
      same-page neighbour, eliminating ``chunk_too_small`` retrieval failures.
    * Globally sequential ``chunk_id`` values are assigned after all per-page
      operations complete.

    Parameters
    ----------
    pages : list[PageRecord]
        Output of :func:`pdf_loader.extract_pages`.
    chunk_size : int
        Target words per chunk (default 400 for the final Task 1 index).
    overlap : int
        Words repeated between consecutive chunks (default 50).
    min_chunk_words : int
        Chunks with fewer than this many words are merged with a neighbour
        (default 60).  Pass ``0`` to disable merging.

    Returns
    -------
    list[Chunk]
        Improved chunks with globally sequential ``chunk_id`` values.
    """
    raw: list[Chunk] = []
    for page in pages:
        page_chunks = chunk_text(
            page["text"],
            chunk_size=chunk_size,
            overlap=overlap,
            source=page["source"],
        )
        for c in page_chunks:
            c["page_number"] = page["page_number"]
            c["ocr"] = page["ocr"]

        if min_chunk_words > 0:
            page_chunks = _merge_short_chunks(page_chunks, min_words=min_chunk_words)

        raw.extend(page_chunks)

    # Reassign globally sequential ids after merging may have collapsed some
    for global_id, chunk in enumerate(raw):
        chunk["chunk_id"] = global_id

    return raw
