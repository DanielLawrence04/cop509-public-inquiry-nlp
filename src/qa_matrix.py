"""
Search QA Matrix evaluation harness.

Loads ALL PDFs in data/raw/ (both recommendation and response documents),
builds a unified corpus using the same chunk_pages pipeline and parameters
as the live notebook (200-word chunks, 30-word overlap), runs multi-mode
search for every query in the bank, auto-labels each result using document
provenance, keyword coverage and optional anchor matching, computes
retrieval metrics, and returns three DataFrames for direct notebook display.

Usage
-----
>>> from pathlib import Path
>>> from src.qa_matrix import run_qa_matrix
>>> queries_df, results_df, metrics_df = run_qa_matrix(
...     data_dir=Path("data/raw"),
...     query_bank_path=Path("data/ground_truth/qa_matrix_queries.json"),
... )
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .chunking import Chunk, chunk_pages, chunk_pages_v2
from .pdf_loader import extract_pages
from .search import SearchResult, keyword_search
from .utils import normalize_text

logger = logging.getLogger(__name__)

# Match notebook defaults exactly
DEFAULT_CHUNK_SIZE = 200
DEFAULT_OVERLAP = 30
DEFAULT_TOP_K = 5

# Auto-label thresholds
_KW_RELEVANT = 0.50   # ≥50% keywords → "likely_relevant"
_KW_PARTIAL = 0.25    # ≥25% keywords → "partial"

# Error-category word-count bounds
_SHORT_CHUNK_WORDS = 60
_LONG_CHUNK_WORDS = 450


# ---------------------------------------------------------------------------
# Corpus construction
# ---------------------------------------------------------------------------

def discover_pdfs(data_dir: Path) -> list[Path]:
    """Return all PDFs in *data_dir*, sorted by name."""
    return sorted(data_dir.glob("*.pdf"))


def build_qa_corpus(
    pdf_paths: list[Path],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    deduplicate: bool = True,
) -> list[Chunk]:
    """
    Build a globally-numbered search corpus from *pdf_paths*.

    Uses the identical ``chunk_pages`` pipeline as the live notebook so that
    evaluation results reflect the real search system rather than a separate
    preprocessing path.

    Parameters
    ----------
    pdf_paths : list[Path]
        PDFs to load (both recommendations and responses).
    chunk_size : int
        Words per chunk — must match the live app (default 200).
    overlap : int
        Overlap words — must match the live app (default 30).
    deduplicate : bool
        If True (default), remove exact-duplicate chunks produced by
        redundant page extractions.  Overlap chunks that merely *share*
        words are kept; only chunks whose normalised text is identical are
        removed.
    """
    corpus: list[Chunk] = []
    next_id = 0
    for path in pdf_paths:
        pages = extract_pages(path)
        page_chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        for chunk in page_chunks:
            chunk["chunk_id"] = next_id
            corpus.append(chunk)
            next_id += 1

    if deduplicate:
        corpus, n_removed = deduplicate_corpus(corpus)
        if n_removed:
            logger.info("Removed %d exact-duplicate chunk(s)", n_removed)

    logger.info("Built QA corpus: %d PDFs, %d chunks total", len(pdf_paths), len(corpus))
    return corpus


def build_qa_corpus_v2(
    pdf_paths: list[Path],
    chunk_size: int = 400,
    overlap: int = 50,
    min_chunk_words: int = 60,
    deduplicate: bool = True,
) -> list[Chunk]:
    """
    Improved corpus builder using :func:`chunk_pages_v2`.

    Identical to :func:`build_qa_corpus` except it uses the improved chunker:
    larger windows (default 400 words), overlap 50, and short-chunk merging.

    Parameters
    ----------
    pdf_paths : list[Path]
        PDFs to load.
    chunk_size : int
        Target words per chunk (default 400 for the final Task 1 index).
    overlap : int
        Overlap words between chunks (default 50).
    min_chunk_words : int
        Chunks below this word count are merged with a neighbour (default 60).
        Pass ``0`` to disable merging.
    deduplicate : bool
        Remove exact-duplicate chunks (default True).
    """
    corpus: list[Chunk] = []
    next_id = 0
    for path in pdf_paths:
        pages = extract_pages(path)
        page_chunks = chunk_pages_v2(
            pages, chunk_size=chunk_size, overlap=overlap,
            min_chunk_words=min_chunk_words,
        )
        for chunk in page_chunks:
            chunk["chunk_id"] = next_id
            corpus.append(chunk)
            next_id += 1

    if deduplicate:
        corpus, n_removed = deduplicate_corpus(corpus)
        if n_removed:
            logger.info("Removed %d exact-duplicate chunk(s) (v2 corpus)", n_removed)

    logger.info(
        "Built QA v2 corpus: %d PDFs, %d chunks total (chunk_size=%d)",
        len(pdf_paths), len(corpus), chunk_size,
    )
    return corpus


def deduplicate_corpus(corpus: list[Chunk]) -> tuple[list[Chunk], int]:
    """
    Remove chunks whose normalised text is identical to a previously seen chunk.

    This guards against exact duplicates that can arise when a PDF page is
    extracted by both the text and OCR paths, or when the same document
    appears in the input list twice.

    Overlap chunks (which *share* a trailing/leading window of words with
    their neighbours) are kept — they contain different text overall.

    Returns
    -------
    deduped : list[Chunk]
        Corpus with duplicates removed (order preserved).
    n_removed : int
        Number of chunks that were dropped.
    """
    seen: set[str] = set()
    deduped: list[Chunk] = []
    for chunk in corpus:
        key = normalize_text(chunk["text"])
        if key not in seen:
            seen.add(key)
            deduped.append(chunk)
    n_removed = len(corpus) - len(deduped)
    return deduped, n_removed


# ---------------------------------------------------------------------------
# Query bank
# ---------------------------------------------------------------------------

def load_query_bank(path: Path) -> list[dict]:
    """Load the QA query bank from a JSON file."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def queries_to_dataframe(queries: list[dict]) -> pd.DataFrame:
    """Return a tidy summary DataFrame of the query bank."""
    rows = []
    for q in queries:
        rows.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "query_type": q["query_type"],
            "expected_docs": ", ".join(q.get("expected_docs") or []),
            "expected_topic": q.get("expected_topic", ""),
            "expected_keywords": ", ".join(q.get("expected_keywords") or []),
            "has_anchor": q.get("anchor") is not None,
            "search_modes": ", ".join(q.get("search_modes", ["keyword"])),
            "top_k": q.get("top_k", DEFAULT_TOP_K),
            "notes": q.get("notes", ""),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Auto-labelling
# ---------------------------------------------------------------------------

def _auto_label(result: SearchResult, query: dict) -> dict:
    """
    Compute three automatic relevance signals for one (result, query) pair.

    Returns a dict with keys:
      doc_match        — bool | None: is the result's source an expected doc?
      keyword_coverage — float | None: fraction of expected_keywords present.
      anchor_match     — bool: does the chunk contain the normalised anchor?
      auto_relevance   — str: one of relevant / likely_relevant / partial /
                              doc_match_only / irrelevant / unknown.
    """
    source = result["source"]
    text_lc = result["text"].lower()
    norm_text = normalize_text(result["text"])

    # doc_match
    expected_docs: list[str] = query.get("expected_docs") or []
    doc_match: bool | None = (source in expected_docs) if expected_docs else None

    # keyword_coverage
    expected_kws = [kw.lower() for kw in (query.get("expected_keywords") or [])]
    if expected_kws:
        hits = sum(1 for kw in expected_kws if kw in text_lc)
        keyword_coverage: float | None = hits / len(expected_kws)
    else:
        keyword_coverage = None

    # anchor_match
    anchor: str | None = query.get("anchor")
    anchor_match = False
    if anchor:
        anchor_match = normalize_text(anchor) in norm_text

    # Derive auto_relevance
    if anchor_match:
        auto_relevance = "relevant"
    elif doc_match and keyword_coverage is not None and keyword_coverage >= _KW_RELEVANT:
        auto_relevance = "likely_relevant"
    elif doc_match and keyword_coverage is not None and keyword_coverage >= _KW_PARTIAL:
        auto_relevance = "partial"
    elif doc_match:
        auto_relevance = "doc_match_only"
    elif doc_match is False:
        auto_relevance = "irrelevant"
    else:
        auto_relevance = "unknown"

    return {
        "doc_match": doc_match,
        "keyword_coverage": round(keyword_coverage, 3) if keyword_coverage is not None else None,
        "anchor_match": anchor_match,
        "auto_relevance": auto_relevance,
    }


def _relevance_grade(auto_relevance: str) -> int:
    """Map auto_relevance label to a numeric grade for nDCG computation."""
    return {
        "relevant": 3,
        "likely_relevant": 2,
        "partial": 1,
        "doc_match_only": 1,
        "irrelevant": 0,
        "unknown": 0,
    }.get(auto_relevance, 0)


# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

_ERROR_DESCRIPTIONS: dict[str, str] = {
    "chunk_too_small": "Chunk too short (<60 words) — likely a header, footer, or OCR artefact.",
    "chunk_too_large": "Chunk very long (>450 words) — relevant signal diluted by surrounding content.",
    "wrong_document": "Wrong document; no topical overlap with expected docs.",
    "wrong_doc_similar_topic": "Wrong document but high score — vocabulary overlap across inquiry reports.",
    "paraphrase_miss": "Correct document but keywords absent — query wording differs from passage.",
    "ranking_issue": "Relevant passage exists in corpus but ranked below position 1.",
    "keyword_coverage_low": "Correct document; expected keywords only partially covered.",
    "other": "Uncategorised failure.",
}


def classify_error(row: dict, query: dict) -> str:
    """
    Assign an error category to a result that is not clearly relevant.

    Only called on rank-1 results whose auto_relevance is not
    'relevant' or 'likely_relevant'.
    """
    auto_rel = row.get("auto_relevance", "unknown")
    if auto_rel in ("relevant", "likely_relevant"):
        return ""

    preview_words = len(str(row.get("returned_preview", "")).split())
    doc_match = row.get("doc_match")
    kw_cov: float = row.get("keyword_coverage") or 0.0
    score: float = row.get("score", 0.0)

    if preview_words < _SHORT_CHUNK_WORDS:
        return "chunk_too_small"
    if preview_words > _LONG_CHUNK_WORDS:
        return "chunk_too_large"
    if not doc_match and score > 0.15:
        return "wrong_doc_similar_topic"
    if not doc_match:
        return "wrong_document"
    if doc_match and kw_cov < 0.20:
        return "paraphrase_miss"
    if doc_match and kw_cov >= 0.20:
        return "ranking_issue"
    return "other"


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_qa_matrix(
    data_dir: Path,
    query_bank_path: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    top_k: int = DEFAULT_TOP_K,
    modes: Optional[list[str]] = None,
    prebuilt_corpus: Optional[list[Chunk]] = None,
    prebuilt_embeddings: Optional[np.ndarray] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run the full QA matrix evaluation and return three DataFrames.

    Parameters
    ----------
    data_dir : Path
        Directory containing raw PDF files (both recommendations and responses).
    query_bank_path : Path
        Path to ``qa_matrix_queries.json``.
    chunk_size : int
        Words per chunk — must match the live notebook (default 200).
    overlap : int
        Overlap words — must match the live notebook (default 30).
    top_k : int
        Number of results retrieved per query/mode.
    modes : list[str], optional
        Search modes to run: "keyword", "semantic", and/or "hybrid".
        Defaults to ["keyword"].  Semantic and hybrid modes are silently
        skipped if ``sentence-transformers`` is not installed.  Hybrid
        mode runs for all queries regardless of query-level ``search_modes``
        settings (it is a meta-strategy that overrides per-query defaults).
    prebuilt_corpus : list[Chunk], optional
        Re-use an already-built corpus to avoid re-loading PDFs.
    prebuilt_embeddings : np.ndarray, optional
        Re-use pre-computed sentence embeddings for semantic mode.

    Returns
    -------
    queries_df : pd.DataFrame
        One row per query — shows the full bank at a glance.
    results_df : pd.DataFrame
        Long-format table: one row per (query × mode × rank position).
        Includes auto-relevance labels and a blank ``manual_relevance``
        column ready for human grading.
    metrics_df : pd.DataFrame
        Aggregate retrieval metrics by (mode, query_type) plus an "ALL"
        row per mode.
    """
    if modes is None:
        modes = ["keyword"]

    # --- Build corpus ---
    if prebuilt_corpus is not None:
        corpus = prebuilt_corpus
        logger.info("Reusing prebuilt corpus (%d chunks)", len(corpus))
    else:
        pdf_paths = discover_pdfs(data_dir)
        if not pdf_paths:
            raise FileNotFoundError(f"No PDF files found in {data_dir}")
        corpus = build_qa_corpus(pdf_paths, chunk_size=chunk_size, overlap=overlap)

    # --- Build embeddings for semantic / hybrid modes ---
    embeddings: Optional[np.ndarray] = None
    active_modes = list(modes)
    needs_embeddings = any(m in active_modes for m in ("semantic", "hybrid"))
    if needs_embeddings:
        if prebuilt_embeddings is not None:
            embeddings = prebuilt_embeddings
        else:
            try:
                from .search import build_embeddings
                logger.info("Encoding %d chunks for dense retrieval…", len(corpus))
                embeddings = build_embeddings(corpus)
                logger.info("Embeddings ready: %s", embeddings.shape)
            except ImportError:
                warnings.warn(
                    "sentence-transformers not installed; semantic and hybrid modes skipped.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                active_modes = [m for m in active_modes if m not in ("semantic", "hybrid")]

    # --- Load query bank ---
    queries = load_query_bank(query_bank_path)
    queries_df = queries_to_dataframe(queries)

    # --- Execute matrix ---
    result_rows: list[dict] = []

    for query in queries:
        # keyword / semantic: respect per-query search_modes setting
        # hybrid: always run when in active_modes (it is a meta-strategy)
        q_declared_modes = set(query.get("search_modes", ["keyword"]))
        q_top_k = query.get("top_k", top_k)

        for mode in active_modes:
            if mode not in ("hybrid",) and mode not in q_declared_modes:
                continue  # skip if query doesn't declare this mode

            if mode == "keyword":
                hits: list[SearchResult] = keyword_search(query["query"], corpus, top_k=q_top_k)
            elif mode == "semantic" and embeddings is not None:
                from .search import semantic_search
                hits = semantic_search(query["query"], corpus, embeddings, top_k=q_top_k)
            elif mode == "hybrid" and embeddings is not None:
                from .search import hybrid_search
                hits = hybrid_search(query["query"], corpus, embeddings, top_k=q_top_k)
            else:
                continue

            for rank, hit in enumerate(hits, start=1):
                labels = _auto_label(hit, query)
                preview = hit["text"][:200].replace("\n", " ")
                result_rows.append({
                    "query_id": query["query_id"],
                    "query": query["query"],
                    "query_type": query["query_type"],
                    "mode": mode,
                    "top_k": q_top_k,
                    "rank": rank,
                    "returned_source": hit["source"],
                    "returned_page": hit["page_number"],
                    "returned_chunk_id": hit["chunk_id"],
                    "returned_preview": preview,
                    "score": round(hit["score"], 4),
                    **labels,
                    "manual_relevance": "",
                    "error_category": "",
                    "notes": "",
                })

    results_df = pd.DataFrame(result_rows)

    # --- Assign error categories to top-1 failures ---
    if not results_df.empty:
        top1_idx = results_df[results_df["rank"] == 1].index
        for idx in top1_idx:
            row = results_df.loc[idx]
            if row["auto_relevance"] not in ("relevant", "likely_relevant"):
                q = next((q for q in queries if q["query_id"] == row["query_id"]), {})
                results_df.at[idx, "error_category"] = classify_error(row.to_dict(), q)

    # --- Compute metrics ---
    metrics_df = _compute_metrics(results_df, queries)

    return queries_df, results_df, metrics_df


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _is_hit(auto_relevance: str) -> bool:
    """True if a result counts as a retrieval hit."""
    return auto_relevance in ("relevant", "likely_relevant")


def _ndcg_at_k(grades: list[int], k: int = 5) -> float:
    """Compute nDCG@k for a single ranked list of relevance grades."""
    def dcg(g: list[int]) -> float:
        return sum(rel / np.log2(i + 2) for i, rel in enumerate(g))
    ideal = sorted(grades, reverse=True)
    idcg = dcg(ideal[:k])
    return dcg(grades[:k]) / idcg if idcg > 0 else 0.0


def _compute_metrics(results_df: pd.DataFrame, queries: list[dict]) -> pd.DataFrame:
    """Return metrics aggregated by (mode, query_type) and overall per mode."""
    if results_df.empty:
        return pd.DataFrame()

    query_lookup = {q["query_id"]: q for q in queries}
    metric_rows: list[dict] = []

    def _row(df_slice: pd.DataFrame, mode: str, qtype: str) -> dict | None:
        qids = df_slice["query_id"].unique()
        n = len(qids)
        if n == 0:
            return None

        r1 = r3 = r5 = p1 = p3 = p5 = top1 = doc1 = mrr_sum = ndcg_sum = 0.0

        for qid in qids:
            qdf = df_slice[df_slice["query_id"] == qid].sort_values("rank")
            hits = [_is_hit(v) for v in qdf["auto_relevance"]]
            grades = [_relevance_grade(v) for v in qdf["auto_relevance"]]
            expected_docs = query_lookup.get(qid, {}).get("expected_docs") or []

            r1 += int(any(hits[:1]))
            r3 += int(any(hits[:3]))
            r5 += int(any(hits[:5]))

            p1 += sum(hits[:1]) / 1
            p3 += sum(hits[:3]) / 3
            p5 += sum(hits[:5]) / max(len(hits[:5]), 1)

            first = next((i + 1 for i, h in enumerate(hits) if h), None)
            mrr_sum += (1.0 / first) if first else 0.0

            top1 += int(bool(hits) and hits[0])

            if not qdf.empty and expected_docs:
                doc1 += int(qdf.iloc[0]["returned_source"] in expected_docs)

            ndcg_sum += _ndcg_at_k(grades, k=5)

        return {
            "mode": mode,
            "query_type": qtype,
            "n_queries": n,
            "recall@1": round(r1 / n, 3),
            "recall@3": round(r3 / n, 3),
            "recall@5": round(r5 / n, 3),
            "precision@1": round(p1 / n, 3),
            "precision@3": round(p3 / n, 3),
            "precision@5": round(p5 / n, 3),
            "mrr": round(mrr_sum / n, 3),
            "top1_accuracy": round(top1 / n, 3),
            "doc_accuracy": round(doc1 / n, 3),
            "ndcg@5": round(ndcg_sum / n, 3),
        }

    for mode in sorted(results_df["mode"].unique()):
        mode_df = results_df[results_df["mode"] == mode]

        overall = _row(mode_df, mode, "ALL")
        if overall:
            metric_rows.append(overall)

        for qtype in sorted(mode_df["query_type"].unique()):
            row = _row(mode_df[mode_df["query_type"] == qtype], mode, qtype)
            if row:
                metric_rows.append(row)

    return pd.DataFrame(metric_rows)


# ---------------------------------------------------------------------------
# Convenience views
# ---------------------------------------------------------------------------

def top1_results_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a compact table with the top-1 result per (query_id, mode),
    plus auto-relevance label and error category. Useful for quick inspection.
    """
    top1 = results_df[results_df["rank"] == 1].copy()
    return top1[[
        "query_id", "query_type", "mode", "query",
        "returned_source", "returned_page", "score",
        "doc_match", "keyword_coverage", "anchor_match",
        "auto_relevance", "error_category",
        "returned_preview",
    ]].reset_index(drop=True)


def failures_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the top-1 rows that are NOT relevant or likely_relevant.
    Useful for debugging the worst-performing queries.
    """
    top1 = results_df[results_df["rank"] == 1]
    return top1[~top1["auto_relevance"].isin(["relevant", "likely_relevant"])].copy().reset_index(drop=True)
