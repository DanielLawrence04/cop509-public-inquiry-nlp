"""
Search functions: TF-IDF keyword search and optional semantic search.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
from typing import TypedDict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk
from .utils import normalize_text


SEMANTIC_SEARCH_VERSION = "2026-05-02-idf-coverage-v3"


def _debug_enabled() -> bool:
    return os.environ.get("DEBUG_SEARCH", "").lower() in ("1", "true", "yes", "on")


def _dlog(msg: str) -> None:
    if _debug_enabled():
        print(f"[DEBUG_SEARCH] {msg}", file=sys.stderr, flush=True)


# Lightweight stop-word list mirrors the one used in the frontend
# `filterQueryTerms` helper so the lexical-coverage signal counts the same
# meaningful tokens the UI surfaces in "Matched: ..." chips.
_LEX_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "there", "this", "to", "was", "were", "will", "with",
    "about", "into", "after", "before",
})


def _meaningful_query_terms(query: str) -> list[str]:
    """Return de-duplicated, lower-cased non-stopword terms with len >= 3."""
    seen: set[str] = set()
    out: list[str] = []
    for tok in re.split(r"\W+", query.lower()):
        if len(tok) < 3 or tok in _LEX_STOPWORDS or tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
    return out


def chunks_fingerprint(chunks: list[Chunk]) -> str:
    """
    Return a short, stable fingerprint of *chunks* used to detect stale
    embedding caches. Two chunk lists with the same fingerprint are treated
    as the same corpus for caching purposes.
    """
    if not chunks:
        return "0::empty"
    head = chunks[0]
    tail = chunks[-1]
    middle = chunks[len(chunks) // 2]
    sig_src = "|".join([
        str(len(chunks)),
        f"{head['source']}#{head['chunk_id']}#{len(head['text'])}",
        f"{middle['source']}#{middle['chunk_id']}#{len(middle['text'])}",
        f"{tail['source']}#{tail['chunk_id']}#{len(tail['text'])}",
    ])
    return f"{len(chunks)}::{hashlib.sha1(sig_src.encode('utf-8')).hexdigest()[:12]}"

# ---------------------------------------------------------------------------
# Structural-chunk detection
# Prevents exact-match contents/appendix chunks from dominating top results
# while preserving recall.
# ---------------------------------------------------------------------------

# Known navigational/structural section headings.
_STRUCTURAL_HEADERS = re.compile(
    r'\b(table\s+of\s+contents?|list\s+of\s+(recommendations?|figures?|tables?|appendices|appendix)'
    r'|appendix|appendices|references?|bibliography|abbreviations?|glossary|index)\b',
    re.IGNORECASE,
)

# Contents-page style line: some text followed by a standalone page number.
# e.g. "Chapter 2: Background ........... 12"  or  "Recommendation 7  42"
_PAGE_REF_LINE = re.compile(r'^.{3,60}\s+\d{1,3}\s*$')

# Numbered heading prefix: "132. TEXT", "7) TEXT", or "A. TEXT"
_NUMBERED_HEADING_PREFIX = re.compile(r'^\s*(?:\d+|[A-Z])[\.\)]\s+\S')

# Score multiplier applied to chunks identified as structural/navigational.
_STRUCTURAL_PENALTY = 0.65

# Stronger penalty for contents/table-of-contents chunks: these often surface
# many query terms (section titles mention key concepts) but contain no
# substantive passage text, so they need a heavier down-weight.
_CONTENTS_PENALTY = 0.45

# Detects an explicit "Contents" header at the start of a chunk.
_CONTENTS_HEADER = re.compile(r'^\s*contents\b', re.IGNORECASE)

# A contents-style line: text followed by a bare page number, or dotted leader.
_CONTENTS_LINE = re.compile(r'^.{3,80}(?:\.{3,}|\s{2,})\s*\d{1,3}\s*$')


def _is_title_only_chunk(text: str) -> bool:
    """
    Return True if *text* is a bare heading or title rather than prose —
    e.g. "132. THE SPACE ECONOMY: ACT NOW OR LOSE OUT".

    Requires all three of:
      - fewer than 35 words
      - no sentence-ending punctuation in the body text
    And at least two of:
      - starts with a numbered/lettered prefix ("132.", "A.")
      - high uppercase ratio (≥ 50 % of alphabetic characters)
      - short single line (no newlines and fewer than 20 words)

    The two-signal gate keeps the rule conservative so that short but
    meaningful prose fragments are not penalised.
    """
    words = text.split()
    if len(words) >= 35:
        return False

    # Strip any leading numbered prefix before checking for sentence endings,
    # so the dot in "132." is not mistaken for sentence punctuation.
    body = re.sub(r'^\s*\d+[\.\)]\s*', '', text.strip())
    if re.search(r'[.!?](?:\s|$)', body):
        return False  # has sentence structure — treat as prose

    # Supporting signals
    starts_with_number = bool(_NUMBERED_HEADING_PREFIX.match(text.strip()))

    alpha = [c for c in text if c.isalpha()]
    high_uppercase = (sum(1 for c in alpha if c.isupper()) / max(len(alpha), 1)) >= 0.5

    is_short_single_line = '\n' not in text.strip() and len(words) < 20

    signal_count = sum([starts_with_number, high_uppercase, is_short_single_line])
    return signal_count >= 2


def _is_contents_chunk(text: str) -> bool:
    """
    Return True if *text* is clearly a contents/table-of-contents chunk.

    Triggers on either:
      - chunk starts with the word "Contents" (with optional leading whitespace), or
      - at least 40 % of non-empty lines look like contents entries
        (text ... page-number or text  page-number).

    Kept deliberately conservative: requires a hard "Contents" lead or a
    high density of contents-style lines to avoid false-positives on
    ordinary lists or short bulleted sections.
    """
    stripped = text.strip()
    if _CONTENTS_HEADER.match(stripped):
        return True

    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if not lines:
        return False
    contents_line_count = sum(1 for ln in lines if _CONTENTS_LINE.match(ln))
    return (contents_line_count / len(lines)) >= 0.4


def _is_structural_chunk(text: str) -> bool:
    """
    Return True if *text* looks like a navigational or structural chunk
    (contents page, appendix, references list, title-only heading, etc.)
    rather than substantive prose.

    Uses three cheap heuristics; requires at least two signals to avoid
    false-positives on legitimate short passages.
    """
    # Title/heading-only check covers short numbered or all-caps headings.
    if _is_title_only_chunk(text):
        return True

    words = text.split()
    word_count = len(words)
    if word_count < 10:
        return False  # too short to analyse reliably

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Signal 1: a recognised structural heading appears in the first 200 chars.
    has_structural_header = bool(_STRUCTURAL_HEADERS.search(text[:200]))

    # Signal 2: high proportion of page-reference lines.
    page_ref_count = sum(1 for ln in lines if _PAGE_REF_LINE.match(ln))
    page_ref_ratio = page_ref_count / max(len(lines), 1)

    # Signal 3: unusually low sentence density (fewer than 1 sentence per 40 words).
    sentence_endings = len(re.findall(r'[.!?](?:\s|$)', text))
    low_sentence_density = (sentence_endings / max(word_count, 1)) < 0.025

    # Two-signal threshold, or an overwhelmingly high page-ref ratio.
    signal_count = sum([has_structural_header, page_ref_ratio > 0.25, low_sentence_density])
    return signal_count >= 2 or page_ref_ratio > 0.5


class SearchResult(TypedDict):
    chunk_id: int
    text: str
    source: str
    page_number: int | None
    score: float


def keyword_search(
    query: str,
    chunks: list[Chunk],
    top_k: int = 5,
) -> list[SearchResult]:
    """
    Rank *chunks* against *query* using TF-IDF cosine similarity.

    Parameters
    ----------
    query : str
        User search query.
    chunks : list[Chunk]
        Corpus to search.
    top_k : int
        Maximum number of results to return.

    Returns
    -------
    list[SearchResult]
        Top-k chunks sorted by descending relevance score.
    """
    if not chunks:
        return []

    texts = [normalize_text(c["text"]) for c in chunks]
    norm_query = normalize_text(query)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([norm_query])

    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Down-weight structural/navigational chunks so prose passages rank higher.
    # Contents pages get a stronger penalty (0.45) because they contain many
    # section-title terms that match queries but carry no substantive text.
    # Other structural chunks (appendices, title-only headings, etc.) get the
    # lighter penalty (0.65); neither type is removed entirely.
    for i, chunk in enumerate(chunks):
        if scores[i] > 0.0:
            text = chunk["text"]
            if _is_contents_chunk(text):
                scores[i] *= _CONTENTS_PENALTY
            elif _is_structural_chunk(text):
                scores[i] *= _STRUCTURAL_PENALTY

    top_indices = np.argsort(scores)[::-1][:top_k]

    results: list[SearchResult] = []
    for idx in top_indices:
        if scores[idx] == 0.0:
            continue
        chunk = chunks[idx]
        results.append(
            SearchResult(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                source=chunk["source"],
                page_number=chunk["page_number"],
                score=float(scores[idx]),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Cross-document (global) search
# ---------------------------------------------------------------------------

def global_search(
    query: str,
    all_chunks: list[Chunk],
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Search across a combined corpus of chunks drawn from multiple documents.

    This is a thin wrapper around :func:`keyword_search` that makes the
    cross-document intent explicit in call sites.  Because each chunk carries
    a ``source`` field (the originating filename), callers can group or label
    results by document without any additional logic.

    Parameters
    ----------
    query : str
        User search query.
    all_chunks : list[Chunk]
        Combined pool of chunks from every loaded document.
    top_k : int
        Maximum number of results to return across *all* documents.

    Returns
    -------
    list[SearchResult]
        Top-k results ranked by TF-IDF cosine similarity.  Each entry
        includes ``source`` (document filename) for clear provenance.
    """
    return keyword_search(query, all_chunks, top_k=top_k)


# ---------------------------------------------------------------------------
# Optional semantic search – only imported when sentence-transformers is available
# ---------------------------------------------------------------------------

# Module-level cache so the SentenceTransformer is loaded once per process.
# Loading the model takes several seconds; constructing it on every call
# would make every semantic query slow even after embeddings are cached.
_ST_MODEL_CACHE: dict[str, object] = {}


def _get_st_model(model_name: str = "all-MiniLM-L6-v2"):
    if model_name not in _ST_MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for semantic search. "
                "Install it with: pip install sentence-transformers"
            ) from exc
        _ST_MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _ST_MODEL_CACHE[model_name]


def build_embeddings(chunks: list[Chunk], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Encode *chunks* into dense embeddings using a sentence-transformer model.

    Parameters
    ----------
    chunks : list[Chunk]
        Chunks to embed.
    model_name : str
        HuggingFace model identifier.

    Returns
    -------
    np.ndarray
        Shape ``(len(chunks), embedding_dim)``. Rows are L2-normalised so
        downstream cosine similarity is a plain dot product.
    """
    model = _get_st_model(model_name)
    texts = [c["text"] for c in chunks]
    embs = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    _dlog(
        f"build_embeddings: {len(chunks)} chunks → "
        f"shape={embs.shape}, fp={chunks_fingerprint(chunks)}"
    )
    return embs


def semantic_search(
    query: str,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    top_k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[SearchResult]:
    """
    Rank *chunks* against *query* using dense semantic similarity.

    Parameters
    ----------
    query : str
        User search query.
    chunks : list[Chunk]
        Corpus (must match the rows in *embeddings*).
    embeddings : np.ndarray
        Pre-computed chunk embeddings from :func:`build_embeddings`.
    top_k : int
        Maximum number of results.
    model_name : str
        Model used to encode the query (must match the corpus model).

    Returns
    -------
    list[SearchResult]
        Top-k chunks sorted by descending cosine similarity.
    """
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"chunks/embeddings mismatch: {len(chunks)} chunks but {embeddings.shape[0]} rows"
        )
    model = _get_st_model(model_name)
    query_vec = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    cosine = cosine_similarity(query_vec, embeddings).flatten()

    # ---- Hybrid lexical rerank --------------------------------------------
    # MiniLM mean-pooling dilutes a short exact phrase across a 500-word
    # chunk; a strong literal hit can score similarly to vague paraphrases
    # of nearby pages. Combine cosine with a lexical-coverage score and an
    # exact-phrase bonus so that exact-phrase queries surface the literal
    # match, while paraphrases still benefit from cosine ranking.
    terms = _meaningful_query_terms(query)
    norm_q = normalize_text(query)
    final = cosine.copy().astype(float)
    coverage = np.zeros(len(chunks), dtype=float)
    phrase_hits = np.zeros(len(chunks), dtype=bool)

    if terms:
        for i, chunk in enumerate(chunks):
            text_lc = chunk["text"].lower()
            hits = sum(1 for t in terms if t in text_lc)
            coverage[i] = hits / len(terms)
            if norm_q and norm_q in normalize_text(chunk["text"]):
                phrase_hits[i] = True

        # Coverage adds up to +0.5 (all meaningful terms present);
        # an exact normalised-phrase substring adds a strong +1.5 so a
        # literal phrase match dominates any vaguely-related fragment.
        final = cosine + 0.5 * coverage + 1.5 * phrase_hits.astype(float)

    # Structural penalties are intentionally NOT applied here. They were
    # hiding legitimate prose chunks whose first 200 chars happened to
    # contain trigger words like "list of recommendations". TF-IDF still
    # uses them; semantic relies on the lexical-coverage signal instead.

    top_indices = np.argsort(final)[::-1][:top_k]

    results: list[SearchResult] = []
    for idx in top_indices:
        chunk = chunks[idx]
        results.append(
            SearchResult(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                source=chunk["source"],
                page_number=chunk["page_number"],
                score=float(final[idx]),
            )
        )

    if _debug_enabled():
        preview = [
            (
                int(idx),
                chunks[idx]["source"],
                chunks[idx]["page_number"],
                chunks[idx]["chunk_id"],
                round(float(cosine[idx]), 4),
                round(float(coverage[idx]), 4),
                bool(phrase_hits[idx]),
                round(float(final[idx]), 4),
            )
            for idx in top_indices[:5]
        ]
        _dlog(
            f"semantic_search[{SEMANTIC_SEARCH_VERSION}]: query={query!r} | "
            f"chunks={len(chunks)} | embs={embeddings.shape} | "
            f"fp={chunks_fingerprint(chunks)} | terms={terms} | "
            f"top5(idx,src,pg,cid,cos,cov,phrase,final)={preview}"
        )

    return results


# ===========================================================================
# Hybrid ranking, query-type detection, document-pair boost
# ===========================================================================

# ---------------------------------------------------------------------------
# Query-type detection
# ---------------------------------------------------------------------------

# Matches "Recommendation 6", "Rec 3", "recommendations 12" etc.
_REC_NUM_RE = re.compile(r'\brec(?:ommendations?)?\s+\d+\b', re.IGNORECASE)

# Numbers that signal a numeric/date query (2+ digit sequences, years, %)
_NUMERIC_RE = re.compile(
    r'\b\d{2,}\b'
    r'|(?:thousand|million|billion|per\s*cent|\d+\s*%|19\d{2}|20\d{2})\b',
    re.IGNORECASE,
)

# Maps query type to recommended keyword/semantic blend weight.
# alpha=1.0 → pure keyword;  alpha=0.0 → pure semantic.
_ALPHA_BY_TYPE: dict[str, float] = {
    "exact_phrase": 0.80,
    "numeric":      0.85,
    "short":        0.40,
    "exploratory":  0.25,
    "general":      0.55,
}


def detect_query_type(query: str) -> str:
    """
    Lightweight classification of *query* into one of five retrieval types.

    Returns
    -------
    "exact_phrase"  — contains a quoted string or numbered recommendation reference
    "numeric"       — contains digits, years, or scale words
    "short"         — 1–2 meaningful tokens after stopword removal
    "exploratory"   — 8+ meaningful tokens
    "general"       — everything else (3–7 tokens)
    """
    if _REC_NUM_RE.search(query) or ('"' in query and query.count('"') >= 2):
        return "exact_phrase"
    if _NUMERIC_RE.search(query):
        return "numeric"
    n = len(_meaningful_query_terms(query))
    if n <= 2:
        return "short"
    if n >= 8:
        return "exploratory"
    return "general"


def recommended_alpha(query: str) -> float:
    """
    Return the recommended keyword/semantic blend weight for *query*.

    alpha=1.0 → pure TF-IDF;  alpha=0.0 → pure semantic (MiniLM).

    +--------------+-------+-----------------------------------------------+
    | Query type   | alpha | Rationale                                     |
    +--------------+-------+-----------------------------------------------+
    | exact_phrase | 0.80  | Exact token match critical; semantics fill gaps|
    | numeric      | 0.85  | Literal numbers handled better by TF-IDF      |
    | short        | 0.40  | Ambiguous; semantic breadth helps             |
    | exploratory  | 0.25  | Long queries encode meaning better as vectors |
    | general      | 0.55  | Slight keyword preference for policy vocab    |
    +--------------+-------+-----------------------------------------------+
    """
    return _ALPHA_BY_TYPE[detect_query_type(query)]


# ---------------------------------------------------------------------------
# Recommendation-heading boost helper (private)
# ---------------------------------------------------------------------------

_REC_HEADING_RE = re.compile(r'\brec(?:ommendations?)?\s+(\d+)\b', re.IGNORECASE)


def _rec_heading_boost_scores(query: str, chunks: list[Chunk]) -> np.ndarray:
    """
    Return a +0.08 bonus array for chunks containing the specific
    recommendation number cited in *query* (e.g. "Recommendation 6").
    Returns zeros for queries without a numbered recommendation reference.
    """
    m = _REC_HEADING_RE.search(query)
    if not m:
        return np.zeros(len(chunks))
    rec_num = m.group(1)
    pat = re.compile(
        rf'\brec(?:ommendations?)?\s+{re.escape(rec_num)}\b', re.IGNORECASE
    )
    bonus = np.zeros(len(chunks))
    for i, chunk in enumerate(chunks):
        if pat.search(chunk["text"]):
            bonus[i] = 0.08
    return bonus


# ---------------------------------------------------------------------------
# Hybrid search
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    top_k: int = 5,
    alpha: float | None = None,
    model_name: str = "all-MiniLM-L6-v2",
    preferred_sources: list[str] | None = None,
    source_boost: float = 0.12,
) -> list[SearchResult]:
    """
    Hybrid TF-IDF + semantic search with adaptive alpha and optional document-pair boost.

    Algorithm
    ---------
    1.  Compute TF-IDF cosine scores with structural-chunk penalties.
    2.  Compute pure semantic cosine (dot product of L2-normalised vectors).
    3.  Normalise TF-IDF to [0, 1] by its maximum; clip semantic to [0, 1].
    4.  Combine: ``alpha * tfidf_norm + (1 - alpha) * semantic_norm``.
    5.  Add IDF-weighted query-term coverage bonus (up to +0.45).
        Each non-stopword query term is weighted by its corpus IDF so that
        distinctive/rare terms (e.g. "postmasters") contribute far more than
        generic terms (e.g. "compensation", "scheme").  This prevents a chunk
        that repeats common query words many times from out-scoring a chunk
        that contains the rare, context-bearing query terms.
    6.  Add exact-phrase bonus (+0.15) where the full normalised query appears verbatim.
    7.  Add recommendation-heading bonus (+0.08) if query cites a numbered recommendation.
    8.  Add source-preference bonus (+source_boost) for *preferred_sources* results.
    9.  Return top-k by descending combined score.

    Parameters
    ----------
    query : str
        User search query.
    chunks : list[Chunk]
        Corpus (must correspond row-for-row to *embeddings*).
    embeddings : np.ndarray
        Pre-computed L2-normalised embeddings from :func:`build_embeddings`.
    top_k : int
        Maximum results to return.
    alpha : float | None
        Keyword/semantic blend (0 = pure semantic, 1 = pure keyword).
        ``None`` (default) triggers automatic selection via :func:`recommended_alpha`.
    model_name : str
        SentenceTransformer model (must match the corpus model).
    preferred_sources : list[str] | None
        Optional document filenames to boost (document-pair awareness).
    source_boost : float
        Score bonus for preferred-source results (default 0.12).
    """
    if not chunks:
        return []
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"hybrid_search: chunks/embeddings mismatch "
            f"({len(chunks)} chunks vs {embeddings.shape[0]} rows)"
        )

    effective_alpha = recommended_alpha(query) if alpha is None else float(alpha)

    # ---- TF-IDF component ----
    texts = [normalize_text(c["text"]) for c in chunks]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_mat = vectorizer.fit_transform(texts)
    q_vec = vectorizer.transform([normalize_text(query)])
    tfidf_raw = cosine_similarity(q_vec, tfidf_mat).flatten()

    for i, chunk in enumerate(chunks):
        if tfidf_raw[i] > 0.0:
            text = chunk["text"]
            if _is_contents_chunk(text):
                tfidf_raw[i] *= _CONTENTS_PENALTY
            elif _is_structural_chunk(text):
                tfidf_raw[i] *= _STRUCTURAL_PENALTY

    tfidf_norm = tfidf_raw / (tfidf_raw.max() + 1e-9)

    # ---- Semantic component (pure cosine, no lexical rerank) ----
    model = _get_st_model(model_name)
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    sem_norm = np.clip(cosine_similarity(q_emb, embeddings).flatten(), 0.0, 1.0)

    # ---- Combine ----
    combined = effective_alpha * tfidf_norm + (1.0 - effective_alpha) * sem_norm

    # ---- IDF-weighted query-term coverage ----
    # Weights each non-stopword query term by its corpus IDF value, then adds a
    # bonus proportional to the fraction of total IDF weight covered in each
    # chunk.  Rare terms (high IDF) dominate; generic terms (low IDF) barely
    # move the score.  Maximum bonus is 0.45 (all distinctive terms present).
    terms = _meaningful_query_terms(query)
    if terms:
        feature_names = vectorizer.get_feature_names_out()
        vocab_lookup: dict[str, int] = {t: i for i, t in enumerate(feature_names)}
        idf_arr = vectorizer.idf_
        idf_max = float(idf_arr.max())
        term_idf_pairs = [
            (t, idf_arr[vocab_lookup[t]] if t in vocab_lookup else idf_max)
            for t in terms
        ]
        total_idf = sum(w for _, w in term_idf_pairs) or 1.0
        idf_coverage = np.zeros(len(chunks), dtype=float)
        for i, chunk in enumerate(chunks):
            text_lc = chunk["text"].lower()
            idf_coverage[i] = sum(w for t, w in term_idf_pairs if t in text_lc) / total_idf
        combined += 0.45 * idf_coverage

    # ---- Exact-phrase bonus ----
    norm_q = normalize_text(query)
    if norm_q:
        for i, chunk in enumerate(chunks):
            if norm_q in normalize_text(chunk["text"]):
                combined[i] += 0.15

    # ---- Recommendation-heading bonus ----
    combined += _rec_heading_boost_scores(query, chunks)

    # ---- Document-pair boost ----
    if preferred_sources:
        for i, chunk in enumerate(chunks):
            if chunk["source"] in preferred_sources:
                combined[i] += source_boost

    top_indices = np.argsort(combined)[::-1][:top_k]
    results: list[SearchResult] = []
    for idx in top_indices:
        chunk = chunks[idx]
        results.append(
            SearchResult(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                source=chunk["source"],
                page_number=chunk["page_number"],
                score=float(combined[idx]),
            )
        )

    _dlog(
        f"hybrid_search: query={query!r} alpha={effective_alpha:.2f} "
        f"(type={detect_query_type(query)}) chunks={len(chunks)} "
        f"top1={results[0]['source'] if results else 'none'}"
    )
    return results


# ---------------------------------------------------------------------------
# Document-pair boost (post-processing utility)
# ---------------------------------------------------------------------------

def doc_pair_boost(
    results: list[SearchResult],
    preferred_sources: list[str],
    boost: float = 0.12,
) -> list[SearchResult]:
    """
    Re-rank *results* by adding *boost* to scores from *preferred_sources*.

    Provides document-pair awareness as a post-processing step compatible with
    any retrieval function.  The conservative default boost ensures that a
    weakly relevant preferred-source result does not outrank a clearly relevant
    result from another document.

    Parameters
    ----------
    results : list[SearchResult]
        Ranked output from any search function.
    preferred_sources : list[str]
        Filenames of the user's selected document pair.
    boost : float
        Score delta applied to preferred-source results (default 0.12).

    Returns
    -------
    list[SearchResult]
        Re-sorted results with boosted preferred-source scores.
    """
    if not preferred_sources:
        return results
    boosted = [
        {**r, "score": r["score"] + boost} if r["source"] in preferred_sources else r
        for r in results
    ]
    return sorted(boosted, key=lambda r: r["score"], reverse=True)


# ===========================================================================
# Context expansion (display utility)
# ===========================================================================

def expand_with_context(
    results: list[SearchResult],
    corpus: list[Chunk],
    window: int = 1,
) -> list[dict]:
    """
    Augment each result with neighbouring chunk text for richer display.

    For each result, looks up the adjacent chunk(s) within the same source
    document and page neighbourhood, and returns up to *window* chunks before
    and after the matched chunk.  Neighbour text is trimmed to 300 characters
    so that display cards remain compact.

    This is a **display-only** utility — it does not affect retrieval scores or
    ranking.  The expanded text is surfaced in the search widget's expandable
    context panel.

    Parameters
    ----------
    results : list[SearchResult]
        Ranked retrieval results (from any search function).
    corpus : list[Chunk]
        The full chunk corpus used to retrieve *results*.
    window : int
        Number of neighbouring chunks to include on each side (default 1).

    Returns
    -------
    list[dict]
        Copies of each result dict with two extra keys:
        ``context_before`` — trailing 300 chars of the preceding chunk (same source),
        ``context_after``  — leading 300 chars of the following chunk (same source).
        Either may be ``None`` if no same-source neighbour exists.
    """
    # Build per-source index: source → list of chunks sorted by chunk_id
    source_index: dict[str, list[Chunk]] = {}
    for chunk in corpus:
        source_index.setdefault(chunk["source"], []).append(chunk)
    for src in source_index:
        source_index[src].sort(key=lambda c: c["chunk_id"])

    expanded: list[dict] = []
    for result in results:
        src = result["source"]
        cid = result["chunk_id"]
        src_chunks = source_index.get(src, [])

        # Find position of this chunk in the source list
        pos: int | None = None
        for j, c in enumerate(src_chunks):
            if c["chunk_id"] == cid:
                pos = j
                break

        context_before: str | None = None
        context_after: str | None = None

        if pos is not None:
            before_idx = pos - window
            after_idx = pos + window
            if before_idx >= 0:
                context_before = src_chunks[before_idx]["text"][-300:]
            if after_idx < len(src_chunks):
                context_after = src_chunks[after_idx]["text"][:300]

        expanded.append({**result, "context_before": context_before, "context_after": context_after})

    return expanded
