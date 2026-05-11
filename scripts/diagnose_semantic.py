"""
Side-by-side diagnostic for the semantic-search regression.

For the natural-clearers query, prints raw cosine, lexical coverage,
phrase hit, final score, and the position in the descending-final
ranking for two reference chunks:

  - the EXPECTED chunk: contains "disagreement as to the approach to
    natural clearers"
  - the WRONG chunk that was previously surfaced: contains "purported
    distinction was" and "wholly untenable"

Usage:
    python -m scripts.diagnose_semantic

Set DEBUG_SEARCH=1 for the in-search trace.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sklearn.metrics.pairwise import cosine_similarity

from src.chunking import chunk_pages
from src.pdf_loader import extract_pages
from src.search import (
    SEMANTIC_SEARCH_VERSION,
    _get_st_model,
    _meaningful_query_terms,
    build_embeddings,
)
from src.utils import normalize_text

QUERY = "There was disagreement as to the approach to natural clearers"
PDF = "Volume_1-Blood-Inquiry-Recomm.pdf"

EXPECTED_NEEDLE = "disagreement as to the approach to natural clearers"
WRONG_NEEDLES = ("purported distinction", "wholly untenable")


def find_chunk(chunks, predicate):
    for i, c in enumerate(chunks):
        if predicate(c):
            return i, c
    return -1, None


def main() -> int:
    print(f"SEMANTIC_SEARCH_VERSION = {SEMANTIC_SEARCH_VERSION}")
    pdf_path = ROOT / "data" / "raw" / PDF
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages)
    print(f"loaded {len(chunks)} chunks from {PDF}")

    embs = build_embeddings(chunks)
    model = _get_st_model()
    qv = model.encode([QUERY], convert_to_numpy=True, normalize_embeddings=True)
    cos = cosine_similarity(qv, embs).flatten()

    terms = _meaningful_query_terms(QUERY)
    norm_q = normalize_text(QUERY)
    coverage = np.zeros(len(chunks), dtype=float)
    phrase = np.zeros(len(chunks), dtype=bool)
    for i, c in enumerate(chunks):
        text_lc = c["text"].lower()
        coverage[i] = sum(1 for t in terms if t in text_lc) / len(terms)
        phrase[i] = norm_q in normalize_text(c["text"])

    final = cos + 0.5 * coverage + 1.5 * phrase.astype(float)
    order = np.argsort(final)[::-1]
    rank_of = {int(idx): r for r, idx in enumerate(order, 1)}

    expected_idx, expected_chunk = find_chunk(
        chunks, lambda c: EXPECTED_NEEDLE in c["text"].lower()
    )
    wrong_idx, wrong_chunk = find_chunk(
        chunks,
        lambda c: all(n in c["text"].lower() for n in WRONG_NEEDLES),
    )

    def show(label, idx, chunk):
        if idx < 0:
            print(f"\n{label}: not found in corpus")
            return
        print(f"\n{label}")
        print(f"  index            : {idx}")
        print(f"  chunk_id         : {chunk['chunk_id']}")
        print(f"  page             : {chunk['page_number']}")
        print(f"  raw cosine       : {cos[idx]:.4f}")
        print(f"  lexical coverage : {coverage[idx]:.4f} (terms={terms})")
        print(f"  exact phrase     : {bool(phrase[idx])}")
        print(f"  final score      : {final[idx]:.4f}")
        print(f"  rank             : {rank_of.get(idx, 'n/a')}")
        snippet = chunk["text"][:200].replace("\n", " ")
        print(f"  text[:200]       : {snippet}…")

    show("EXPECTED chunk (natural clearers passage)", expected_idx, expected_chunk)
    show("WRONG chunk (purported distinction / wholly untenable)", wrong_idx, wrong_chunk)

    print("\nTop 5 by final score:")
    for r, idx in enumerate(order[:5], 1):
        c = chunks[int(idx)]
        snippet = c["text"][:140].replace("\n", " ")
        print(
            f"  #{r} idx={int(idx)} pg={c['page_number']} "
            f"cos={cos[int(idx)]:.4f} cov={coverage[int(idx)]:.4f} "
            f"phrase={bool(phrase[int(idx)])} final={final[int(idx)]:.4f} "
            f":: {snippet}…"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
