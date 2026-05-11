"""
Retrieval evaluation utilities for Task 1 search assessment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, TypedDict

from .chunking import Chunk, chunk_pages
from .pdf_loader import extract_pages
from .search import SearchResult, keyword_search
from .utils import normalize_text


class RelevantPassageSpec(TypedDict):
    source: str
    anchor: str


class RetrievalQuerySpec(TypedDict):
    query_id: str
    query: str
    description: str
    relevant_passages: list[RelevantPassageSpec]


class ResolvedRetrievalQuery(TypedDict):
    query_id: str
    query: str
    description: str
    relevant_chunk_ids: list[int]
    relevant_passages: list[RelevantPassageSpec]
    unresolved_passages: list[RelevantPassageSpec]


class RetrievalQueryMetrics(TypedDict):
    query_id: str
    query: str
    relevant_count: int
    retrieved_count: int
    hits_at_3: int
    hits_at_5: int
    precision_at_3: float
    recall_at_5: float
    mrr: float
    first_relevant_rank: int | None


class RetrievalEvaluationResult(TypedDict):
    summary: dict[str, float]
    per_query: list[RetrievalQueryMetrics]
    runs: dict[str, list[SearchResult]]


def build_search_evaluation_corpus(
    pdf_paths: list[str | Path],
    chunk_size: int = 200,
    overlap: int = 30,
) -> list[Chunk]:
    """
    Build a search corpus with globally unique chunk ids for retrieval evaluation.
    """
    corpus: list[Chunk] = []
    next_chunk_id = 0
    for pdf_path in pdf_paths:
        pages = extract_pages(pdf_path)
        page_chunks = chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
        for chunk in page_chunks:
            corpus.append(
                Chunk(
                    chunk_id=next_chunk_id,
                    text=chunk["text"],
                    source=chunk["source"],
                    page_number=chunk["page_number"],
                    ocr=chunk["ocr"],
                )
            )
            next_chunk_id += 1
    return corpus


def load_retrieval_queries(path: str | Path) -> list[RetrievalQuerySpec]:
    """Load retrieval query specifications from JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_retrieval_queries(
    queries: list[RetrievalQuerySpec],
    chunks: list[Chunk],
) -> list[ResolvedRetrievalQuery]:
    """
    Resolve source/anchor relevance judgements to concrete chunk ids.
    """
    resolved: list[ResolvedRetrievalQuery] = []
    for spec in queries:
        relevant_ids: set[int] = set()
        unresolved: list[RelevantPassageSpec] = []

        for passage in spec["relevant_passages"]:
            source = passage["source"]
            anchor = normalize_text(passage["anchor"])
            matches = [
                chunk["chunk_id"]
                for chunk in chunks
                if chunk["source"] == source and anchor in normalize_text(chunk["text"])
            ]
            if matches:
                relevant_ids.update(matches)
            else:
                unresolved.append(passage)

        resolved.append(
            ResolvedRetrievalQuery(
                query_id=spec["query_id"],
                query=spec["query"],
                description=spec["description"],
                relevant_chunk_ids=sorted(relevant_ids),
                relevant_passages=spec["relevant_passages"],
                unresolved_passages=unresolved,
            )
        )
    return resolved


def evaluate_retrieval(
    queries: list[ResolvedRetrievalQuery],
    chunks: list[Chunk],
    *,
    top_k: int = 5,
    search_fn: Callable[[str, list[Chunk], int], list[SearchResult]] = keyword_search,
) -> RetrievalEvaluationResult:
    """
    Evaluate retrieval quality with Precision@3, Recall@5, and MRR.
    """
    if not queries:
        raise ValueError("At least one retrieval query is required.")

    per_query: list[RetrievalQueryMetrics] = []
    runs: dict[str, list[SearchResult]] = {}

    for query in queries:
        relevant = set(query["relevant_chunk_ids"])
        if not relevant:
            raise ValueError(f"Query '{query['query_id']}' has no resolved relevant chunks.")

        hits = search_fn(query["query"], chunks, top_k=top_k)
        runs[query["query_id"]] = hits
        retrieved_ids = [hit["chunk_id"] for hit in hits]

        hits_at_3 = sum(1 for chunk_id in retrieved_ids[:3] if chunk_id in relevant)
        hits_at_5 = sum(1 for chunk_id in retrieved_ids[:5] if chunk_id in relevant)

        first_relevant_rank = next(
            (rank for rank, chunk_id in enumerate(retrieved_ids, start=1) if chunk_id in relevant),
            None,
        )
        reciprocal_rank = 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank

        per_query.append(
            RetrievalQueryMetrics(
                query_id=query["query_id"],
                query=query["query"],
                relevant_count=len(relevant),
                retrieved_count=len(retrieved_ids),
                hits_at_3=hits_at_3,
                hits_at_5=hits_at_5,
                precision_at_3=hits_at_3 / 3.0,
                recall_at_5=hits_at_5 / len(relevant),
                mrr=reciprocal_rank,
                first_relevant_rank=first_relevant_rank,
            )
        )

    summary = {
        "queries": len(per_query),
        "mean_precision_at_3": sum(item["precision_at_3"] for item in per_query) / len(per_query),
        "mean_recall_at_5": sum(item["recall_at_5"] for item in per_query) / len(per_query),
        "mean_mrr": sum(item["mrr"] for item in per_query) / len(per_query),
    }

    return RetrievalEvaluationResult(summary=summary, per_query=per_query, runs=runs)


def retrieval_summary_to_dataframe(result: RetrievalEvaluationResult):
    """Convert summary metrics to a compact dataframe."""
    import pandas as pd

    return pd.DataFrame(
        [
            {"metric": "Queries", "value": int(result["summary"]["queries"])},
            {"metric": "Mean Precision@3", "value": result["summary"]["mean_precision_at_3"]},
            {"metric": "Mean Recall@5", "value": result["summary"]["mean_recall_at_5"]},
            {"metric": "Mean MRR", "value": result["summary"]["mean_mrr"]},
        ]
    )


def retrieval_per_query_to_dataframe(result: RetrievalEvaluationResult):
    """Convert per-query retrieval metrics to a dataframe."""
    import pandas as pd

    return pd.DataFrame(result["per_query"])
