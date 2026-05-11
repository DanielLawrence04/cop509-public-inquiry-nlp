# src package – expose public API for convenient imports
from .pdf_loader import load_pdf_text, extract_pages
from .chunking import chunk_text, chunk_pages, chunk_pages_v2, detect_chunk_heading
from .search import (
    keyword_search, global_search, build_embeddings, semantic_search,
    hybrid_search, detect_query_type, recommended_alpha, expand_with_context,
)
from .extraction import extract_recommendations
from .alignment import match_recommendations_to_responses
from .classification import classify_response, classify_batch
from .evaluation import compare_to_ground_truth, results_to_dataframe
from .retrieval_evaluation import (
    build_search_evaluation_corpus,
    evaluate_retrieval,
    load_retrieval_queries,
    resolve_retrieval_queries,
    retrieval_per_query_to_dataframe,
    retrieval_summary_to_dataframe,
)
from .preview import render_page_with_highlights, PreviewResult
from .utils import clean_text, normalize_text, token_overlap_score, save_json, load_json

__all__ = [
    "load_pdf_text", "extract_pages",
    "chunk_text", "chunk_pages", "chunk_pages_v2", "detect_chunk_heading",
    "keyword_search", "global_search", "build_embeddings", "semantic_search",
    "hybrid_search", "detect_query_type", "recommended_alpha", "expand_with_context",
    "extract_recommendations",
    "match_recommendations_to_responses",
    "classify_response", "classify_batch",
    "compare_to_ground_truth", "results_to_dataframe",
    "build_search_evaluation_corpus",
    "evaluate_retrieval",
    "load_retrieval_queries",
    "resolve_retrieval_queries",
    "retrieval_per_query_to_dataframe",
    "retrieval_summary_to_dataframe",
    "render_page_with_highlights", "PreviewResult",
    "clean_text", "normalize_text", "token_overlap_score", "save_json", "load_json",
]
