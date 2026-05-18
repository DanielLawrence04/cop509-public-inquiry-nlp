"""Search endpoint — TF-IDF, semantic, and hybrid retrieval.

Default retriever is hybrid (adaptive TF-IDF + MiniLM blend).
Falls back gracefully to tfidf when sentence-transformers is unavailable
or ENABLE_SEMANTIC_SEARCH=false (local lightweight mode).

Each result is enriched with:
  matched_terms — non-stopword query tokens present in the chunk text
  confidence    — "high" / "medium" / "low" derived from score thresholds
  heading       — leading section heading detected in the chunk (or None)
  context_before / context_after — neighbouring chunk text (300 chars each)

SearchResponse also carries:
  query_type — detect_query_type() classification of the query
  alpha      — adaptive blend weight (hybrid mode only)
"""
from __future__ import annotations
import asyncio
import os
import re
import sys
import time

from fastapi import APIRouter, HTTPException

from backend.core.state import pipeline, SEMANTIC_AVAILABLE
from backend.core.logger import log
from backend.models.requests import SearchRequest
from backend.models.responses import SearchResponse, SearchResultModel

router = APIRouter()

from src.search import SEMANTIC_SEARCH_VERSION

_LEX_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "there", "this", "to", "was", "were", "will", "with",
    "about", "into", "after", "before",
})


def _debug_search() -> bool:
    return os.environ.get("DEBUG_SEARCH", "").lower() in ("1", "true", "yes", "on")


def _dlog(msg: str) -> None:
    if _debug_search():
        print(f"[DEBUG_SEARCH route] {msg}", file=sys.stderr, flush=True)


if SEMANTIC_AVAILABLE:
    print(
        f"[SEARCH] route loaded | semantic=enabled | SEMANTIC_SEARCH_VERSION={SEMANTIC_SEARCH_VERSION}",
        file=sys.stderr,
        flush=True,
    )
else:
    print(
        f"[SEARCH] route loaded | semantic=disabled (local lightweight mode) "
        f"| lexical search active | SEMANTIC_SEARCH_VERSION={SEMANTIC_SEARCH_VERSION}",
        file=sys.stderr,
        flush=True,
    )

# Clear stale embedding cache on hot-reload.
if pipeline.embeddings_cache:
    print(
        f"[SEARCH] clearing {len(pipeline.embeddings_cache)} stale cache "
        f"entries from previous code version",
        file=sys.stderr,
        flush=True,
    )
    pipeline.embeddings_cache.clear()


# Coalesces concurrent first-search requests so we never run the PDF→chunks
# pipeline more than once for the same cold backend.
_DEFAULT_CORPUS_LOCK = asyncio.Lock()


async def _ensure_default_search_corpus_loaded() -> None:
    """Load the 5 coursework pairs into ``preset_cache`` if the search corpus
    is empty so the local app can search immediately. Idempotent and safe to
    call on every request.

    Raises ``HTTPException(500)`` with a clear, user-facing reason if the
    underlying PDFs are missing or the loader crashes.
    """
    # Fast path: at least one preset has chunked snapshot, OR the active pair
    # already has chunks loaded the legacy way. Nothing to do.
    if pipeline.preset_cache:
        return
    if pipeline.policy_chunks or pipeline.response_chunks:
        return

    async with _DEFAULT_CORPUS_LOCK:
        # Re-check under the lock — another concurrent first-search may have
        # populated the cache while we were waiting.
        if pipeline.preset_cache:
            return
        if pipeline.policy_chunks or pipeline.response_chunks:
            return

        print(
            "[SEARCH] corpus empty; loading default search corpus...",
            file=sys.stderr,
            flush=True,
        )
        log.emit("search.autoload", "corpus empty; loading default search corpus", "info")

        t0 = time.monotonic()
        try:
            pairs_loaded, chunks_loaded = await asyncio.to_thread(
                pipeline.ensure_default_search_corpus
            )
        except FileNotFoundError as exc:
            print(
                f"[SEARCH] default corpus load FAILED: {exc}",
                file=sys.stderr,
                flush=True,
            )
            log.emit("search.autoload", f"PDF missing: {exc}", "err")
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Search corpus could not be prepared because a required "
                    f"local PDF is missing: {exc}"
                ),
            )
        except Exception as exc:
            print(
                f"[SEARCH] default corpus load FAILED: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            log.emit("search.autoload", f"load failed: {exc}", "err")
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Search corpus could not be prepared "
                    f"({type(exc).__name__}: {exc})."
                ),
            )

        ms = int((time.monotonic() - t0) * 1000)
        print(
            f"[SEARCH] default search corpus loaded: {pairs_loaded} pairs, "
            f"{chunks_loaded} chunks ({ms} ms)",
            file=sys.stderr,
            flush=True,
        )
        log.emit(
            "search.autoload",
            f"loaded {pairs_loaded} pairs, {chunks_loaded} chunks",
            "ok",
            ms=ms,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _ensure_embeddings(pid: str, pair_chunks: list):
    """Return embeddings for *pair_chunks*, rebuilding if fingerprint changed."""
    from src.search import build_embeddings, chunks_fingerprint

    fp = chunks_fingerprint(pair_chunks)
    cached = pipeline.embeddings_cache.get(pid)
    if cached is None or not isinstance(cached, tuple) or cached[0] != fp:
        _dlog(f"cache MISS pid={pid} fp={fp} — building {len(pair_chunks)} chunks")
        embs = await asyncio.to_thread(build_embeddings, pair_chunks)
        pipeline.embeddings_cache[pid] = (fp, embs)
        return embs
    _dlog(f"cache HIT  pid={pid} fp={fp} rows={cached[1].shape[0]}")
    return cached[1]


def _confidence(score: float) -> str:
    if score >= 0.50:
        return "high"
    if score >= 0.20:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Search endpoint
# ---------------------------------------------------------------------------

@router.post("/", response_model=SearchResponse)
async def search(body: SearchRequest):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # Downgrade semantic/hybrid to tfidf when sentence-transformers is not
    # available (or ENABLE_SEMANTIC_SEARCH=false in local lightweight mode).
    retriever = body.retriever
    if retriever in ("semantic", "hybrid") and not SEMANTIC_AVAILABLE:
        print(
            f"[SEARCH] downgrading retriever={retriever!r} → 'tfidf' "
            "(optional semantic/hybrid mode unavailable locally; lexical fallback active)",
            file=sys.stderr,
            flush=True,
        )
        retriever = "tfidf"

    needs_embeddings = retriever in ("semantic", "hybrid")

    # Auto-load the local corpus on first search so users can search before
    # manually loading a pair on the Documents tab. Idempotent; the lock inside
    # coalesces concurrent first hits.
    await _ensure_default_search_corpus_loaded()

    # ---- Build chunk pool ------------------------------------------------
    pair_id_map: dict[tuple[int, str], str] = {}

    if body.scope == "all":
        chunks: list = []
        covered_pids: set = set()
        for pid, snapshot in pipeline.preset_cache.items():
            covered_pids.add(pid)
            for c in snapshot.policy_chunks + snapshot.response_chunks:
                pair_id_map[(c["chunk_id"], c["source"])] = pid
                chunks.append(c)
        if (
            (pipeline.policy_chunks or pipeline.response_chunks)
            and pipeline.preset_id not in covered_pids
        ):
            active_pid = pipeline.preset_id or "__active__"
            for c in pipeline.policy_chunks + pipeline.response_chunks:
                pair_id_map[(c["chunk_id"], c["source"])] = active_pid
                chunks.append(c)
        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No documents loaded — load a preset pair first",
            )
    else:  # current
        if not pipeline.policy_chunks and not pipeline.response_chunks:
            raise HTTPException(
                status_code=400,
                detail="No documents loaded — load a preset pair first",
            )
        if pipeline.preset_id is None:
            raise HTTPException(
                status_code=400,
                detail="No active pair selected — load a preset first",
            )
        chunks = pipeline.policy_chunks + pipeline.response_chunks
        pair_id_map = {
            (c["chunk_id"], c["source"]): pipeline.preset_id for c in chunks
        }

    t0 = time.monotonic()

    # ---- Build embeddings if needed -------------------------------------
    embeddings = None
    if needs_embeddings:
        import numpy as np

        if body.scope == "all":
            ordered_chunks: list = []
            all_embs: list = []
            covered_pids2: set = set()
            for pid, snapshot in pipeline.preset_cache.items():
                covered_pids2.add(pid)
                pair_chunks = snapshot.policy_chunks + snapshot.response_chunks
                embs = await _ensure_embeddings(pid, pair_chunks)
                ordered_chunks.extend(pair_chunks)
                all_embs.append(embs)
            if (
                (pipeline.policy_chunks or pipeline.response_chunks)
                and pipeline.preset_id not in covered_pids2
            ):
                active_pid = pipeline.preset_id or "__active__"
                active_chunks = pipeline.policy_chunks + pipeline.response_chunks
                embs = await _ensure_embeddings(active_pid, active_chunks)
                ordered_chunks.extend(active_chunks)
                all_embs.append(embs)
            chunks = ordered_chunks
            embeddings = np.vstack(all_embs)
        else:
            pid = pipeline.preset_id
            embeddings = await _ensure_embeddings(pid, chunks)

        if embeddings.shape[0] != len(chunks):
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Embedding/chunks misalignment: "
                    f"embeddings rows={embeddings.shape[0]}, chunks={len(chunks)}"
                ),
            )
        print(
            f"[SEARCH] {SEMANTIC_SEARCH_VERSION} | query={body.query!r} "
            f"retriever={retriever} scope={body.scope} "
            f"chunks={len(chunks)} embs={embeddings.shape}",
            file=sys.stderr,
            flush=True,
        )

    # ---- Run retrieval ---------------------------------------------------
    if retriever == "tfidf":
        from src.search import keyword_search
        results = keyword_search(body.query, chunks, top_k=body.top_k)

    elif retriever == "semantic":
        from src.search import semantic_search
        results = await asyncio.to_thread(
            semantic_search, body.query, chunks, embeddings, body.top_k
        )

    else:  # hybrid (default)
        from src.search import hybrid_search
        results = await asyncio.to_thread(
            hybrid_search, body.query, chunks, embeddings, body.top_k
        )

    elapsed = int((time.monotonic() - t0) * 1000)

    # ---- Enrich results -------------------------------------------------
    from src.search import expand_with_context, detect_query_type, recommended_alpha
    from src.chunking import detect_chunk_heading

    query_type = detect_query_type(body.query)
    alpha = recommended_alpha(body.query) if retriever == "hybrid" else None

    expanded = expand_with_context(results, chunks, window=1)

    query_tokens = {
        t for t in re.split(r"\W+", body.query.lower())
        if len(t) >= 3 and t not in _LEX_STOPWORDS
    }

    enriched: list[SearchResultModel] = []
    for r in expanded:
        text_lower = r["text"].lower()
        matched = sorted(t for t in query_tokens if t in text_lower)
        enriched.append(SearchResultModel(
            chunk_id=r["chunk_id"],
            text=r["text"],
            source=r["source"],
            page_number=r["page_number"],
            score=r["score"],
            pair_id=pair_id_map.get((r["chunk_id"], r["source"])),
            matched_terms=matched,
            confidence=_confidence(r["score"]),
            heading=detect_chunk_heading(r["text"]),
            context_before=r.get("context_before"),
            context_after=r.get("context_after"),
        ))

    # ---- Log ------------------------------------------------------------
    from src.utils import normalize_text
    query_terms_count = len(normalize_text(body.query).split())
    top_score = results[0]["score"] if results else None

    log_event = {
        "tfidf": "search.keyword_search",
        "semantic": "search.semantic_search",
        "hybrid": "search.hybrid_search",
    }.get(retriever, "search.search")

    log.emit(
        log_event,
        f'("{body.query}", k={body.top_k}, scope={body.scope})',
        "ok",
        ms=elapsed,
    )

    return SearchResponse(
        results=enriched,
        query=body.query,
        retriever=retriever,
        scope=body.scope,
        elapsed_ms=elapsed,
        chunks_searched=len(chunks),
        top_score=top_score,
        query_terms=query_terms_count,
        results_returned=len(enriched),
        query_type=query_type,
        alpha=alpha,
    )
