"""
Sanity check for semantic search.

Ranks the natural-clearers query against the Volume_1 Blood Inquiry
recommendations PDF and asserts:

  - top-1 contains all four exact phrase terms
    (disagreement, approach, natural, clearers)
  - top-3 contains the literal phrase "natural clearers"

Run with:

    python -m scripts.sanity_semantic

Set DEBUG_SEARCH=1 to see the per-call rank/cosine/coverage trace.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.chunking import chunk_pages
from src.pdf_loader import extract_pages
from src.search import build_embeddings, semantic_search

QUERY = "There was disagreement as to the approach to natural clearers"
EXPECTED_DOC = "Volume_1-Blood-Inquiry-Recomm.pdf"
PHRASE_TERMS = ("disagreement", "approach", "natural", "clearers")
PHRASE_LITERAL = "natural clearers"


def main() -> int:
    pdf = ROOT / "data" / "raw" / EXPECTED_DOC
    if not pdf.exists():
        print(f"FAIL: missing PDF at {pdf}")
        return 2

    pages = extract_pages(pdf)
    chunks = chunk_pages(pages)
    print(f"loaded {len(chunks)} chunks from {EXPECTED_DOC}")

    embs = build_embeddings(chunks)
    results = semantic_search(QUERY, chunks, embs, top_k=5)

    print(f"\nQuery: {QUERY!r}")
    print("Top 5 semantic results:")
    for i, r in enumerate(results, 1):
        snippet = r["text"][:140].replace("\n", " ")
        present = [t for t in PHRASE_TERMS if t in r["text"].lower()]
        print(
            f"  #{i} score={r['score']:.4f} src={r['source']} "
            f"pg={r['page_number']} terms={present} :: {snippet}…"
        )

    if not results:
        print("FAIL: no results")
        return 1

    top1_text = results[0]["text"].lower()
    top1_has_all_terms = all(t in top1_text for t in PHRASE_TERMS)
    if not top1_has_all_terms:
        missing = [t for t in PHRASE_TERMS if t not in top1_text]
        print(
            f"FAIL: top-1 missing required phrase terms {missing}. "
            f"Top-1 was {results[0]['source']} pg={results[0]['page_number']}"
        )
        return 1

    top3_has_phrase = any(
        PHRASE_LITERAL in r["text"].lower() for r in results[:3]
    )
    if not top3_has_phrase:
        print(
            f"FAIL: top-3 does not contain literal phrase {PHRASE_LITERAL!r}"
        )
        return 1

    print(
        f"\nPASS: top-1 contains all of {PHRASE_TERMS} and top-3 contains "
        f"'{PHRASE_LITERAL}' on page {results[0]['page_number']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
