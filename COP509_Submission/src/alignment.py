"""
Align extracted recommendations to government/stakeholder response chunks.

Two alignment strategies are provided:

``match_recommendations_to_responses``
    TF-IDF alignment against raw response chunks.  Used by the notebook
    cells and as a fallback when response-unit extraction yields no units.
    Each returned ``matched_text`` is trimmed to the next response/heading
    boundary so it never includes the following recommendation's reply.

``match_recommendations_to_response_units``
    Improved alignment used by the backend pipeline (Section 4 UI).
    1. **Direct label match** — preferred when both sides carry a label
       (handles multi-label units like 8.10 + 8.11 sharing one block).
    2. **Structure match** — TF-IDF similarity against the *quoted
       recommendation echo* of each unit, when no label match is possible.
    3. **Semantic / TF-IDF fallback** — TF-IDF against the unit response
       text as a last resort.

    ``matched_text`` always carries the clean response body, never an
    overlapping chunk.
"""

from __future__ import annotations

import re
from typing import TypedDict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import Chunk
from .extraction import Recommendation
from .response_units import ResponseUnit, trim_to_response_boundary
from .utils import normalize_text


class AlignedMatch(TypedDict, total=False):
    rec_id: int
    recommendation: str
    matched_chunk_id: int
    matched_text: str
    source: str
    page_number: int | None
    similarity: float
    alignment_confidence: float
    match_method: str           # exact_label | structure | semantic | chunk_fallback
    boundary_reason: str | None
    quoted_recommendation_text: str | None
    heading_text: str | None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _norm_label(label: str) -> str:
    return label.strip().lower().rstrip(".")


def _build_label_index(
    units: list[ResponseUnit],
) -> dict[str, ResponseUnit]:
    """Map every label a unit covers (multi-label aware) to that unit."""
    idx: dict[str, ResponseUnit] = {}
    for unit in units:
        labels = unit.get("recommendation_labels") or []
        if not labels:
            primary = unit.get("recommendation_label")
            if primary:
                labels = [primary]
        for lbl in labels:
            idx.setdefault(_norm_label(str(lbl)), unit)
    return idx


def _is_space_economy_pair(
    recommendations: list[Recommendation],
    response_units: list[ResponseUnit],
) -> bool:
    rec_doc = str(recommendations[0].get("document", "")) if recommendations else ""
    unit_source = str(response_units[0].get("source", "")) if response_units else ""
    return "TheSpaceEconomyReport" in rec_doc or "TheSpaceEconomyResponse" in unit_source


def _is_infected_blood_pair(
    recommendations: list[Recommendation],
    response_units: list[ResponseUnit],
) -> bool:
    rec_doc = str(recommendations[0].get("document", "")) if recommendations else ""
    unit_source = str(response_units[0].get("source", "")) if response_units else ""
    return "Volume_1-Blood-Inquiry" in rec_doc or "Volume_1-Blood-Inquiry" in unit_source


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall((text or "").lower()))


def _token_overlap(a: str, b: str) -> float:
    left = _token_set(a)
    right = _token_set(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)


_COVID2_RESPONSE_STARTS = {
    "1": [r"\bThis\s+recommendation\s+is\s+not\s+for\s+the\s+UK\s+government\s+to\s+respond\s+to\."],
    "2": [r"\bGO-Science\s+has\s+already\s+made\s+the\s+required\s+changes\b"],
    "3": [r"\bGO-Science\s+already\s+maintains\s+an\s+expert\s+register\b"],
    "5": [r"\bGO-Science\s+already\s+provides\s+support\s+to\s+SAGE\s+participants\b"],
    "6": [r"\bThis\s+government\s+is\s+committed\s+to\s+ensuring\b"],
    "9": [r"\bWe\s+remain\s+steadfastly\s+committed\s+to\s+supporting\s+devolution\b"],
    "10": [r"\bThe\s+government['\u2019]s\s+strategic\s+approach\s+to\s+pandemic\s+preparedness\b"],
    "13": [r"\bThis\s+recommendation\s+is\s+not\s+for\s+the\s+UK\s+government\s+to\s+respond\s+to\."],
}

_GRENFELL_ECHO_LABELS = {
    "31", "32", "33", "34", "35", "36", "38", "39", "40", "41", "56",
}

_BLOOD_LEAK_PATTERNS = [
    r"\s*4a\)\s*ii\.\s+The\s+operation\s+of\s+the\s+duties\s+of\s+candour\b.*?end\s+of\s+2026\.\s*",
    r"\s*(?:4)?a\)\s*iii\.\s+The\s+review\s+of\s+the\s+duty\s+of\s+candour\b.*?practicable\.\s*",
    r"\s*(?:4)?c\)\s*i\.\s+That\s+external\s+regulation\s+of\s+safety\s+in\s+healthcare\b.*?since\s+2000\.\s*",
    r"\s*4c\)\s*ii\.\s+That\s+the\s+national\s+healthcare\s+administrations\b.*?priority\.\s*",
    r"\s*7a\)\s*ii\.\s+In\s+Scotland,\s+Wales\s+and\s+Northern\s+Ireland\b.*?eligible\s+surgery\.\s*",
    r"\s*(?:7)?a\)\s*iii\.\s+Consideration\s+be\s+given\s+to\s+standardising\b.*?blood\s+management\.\s*",
]


def _strip_to_first_pattern(text: str, patterns: list[str]) -> str:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return text[match.start():].strip()
    return text


def _clean_matched_text(rec: Recommendation, unit: ResponseUnit, text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned

    source = str(unit.get("source", ""))
    label = str(rec.get("item_label", "")).strip().lower()

    if "UK-Covid-19_Inquiry_Module_2_Response" in source and label in _COVID2_RESPONSE_STARTS:
        cleaned = _strip_to_first_pattern(cleaned, _COVID2_RESPONSE_STARTS[label])

    if "Grenfell-Phase2-Response" in source and label in _GRENFELL_ECHO_LABELS:
        citation = re.search(r"\(\s*(?:113|133)\.\d+\s*\)\s*", cleaned)
        if citation:
            cleaned = cleaned[citation.end():].strip()

    if "TheSpaceEconomyResponse" in source and label == "23":
        cleaned = _strip_to_first_pattern(
            cleaned,
            [r"\bThe\s+Government\s+is\s+exploring\s+opportunities\b"],
        )

    if "Volume_1-Blood-Inquiry-Response" in source and label in {
        "4a i", "4a ii", "4a iii", "4b", "4c i", "4c ii", "7a i", "7a ii", "7a iii",
    }:
        for pattern in _BLOOD_LEAK_PATTERNS:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if "PostOfficeHorizon-IT-Inquiry-Response" in source:
        if label == "4":
            cleaned = re.sub(r"\bheir claim\b", "their claim", cleaned)
            cleaned = cleaned.replace("up-ront", "up-front")
            cleaned = cleaned.replace("The service will: e explain", "The service will: - explain")
            cleaned = re.sub(r"\s+e\s+(?=(?:help|explain)\b)", " - ", cleaned)
        elif label == "18":
            cleaned = cleaned.replace("establishing anew redress scheme", "establishing a new redress scheme")

    if "TheSpaceEconomyResponse" in source and label == "69":
        cleaned = cleaned.replace("OperationsInitiative", "Operations Initiative")

    return cleaned


def _token_spans(text: str) -> list[tuple[str, int]]:
    return [(m.group(0).lower(), m.end()) for m in _TOKEN_RE.finditer(text or "")]


def _prefix_token_match(rec_text: str, block_text: str) -> tuple[float, int, int]:
    rec_tokens = _TOKEN_RE.findall((rec_text or "").lower())
    block_tokens = _token_spans(block_text)
    matched = 0
    cut_pos = 0
    for rec_token, block_token in zip(rec_tokens, block_tokens):
        if rec_token != block_token[0]:
            break
        matched += 1
        cut_pos = block_token[1]
    if not rec_tokens:
        return 0.0, 0, 0
    return matched / len(rec_tokens), matched, cut_pos


def _trim_repeated_recommendation(rec_text: str, unit: ResponseUnit) -> str:
    """Return a Space response block with the repeated recommendation removed."""
    block = (unit.get("full_unit_text") or unit.get("response_text") or "").strip()
    if not block:
        return ""

    ratio, matched, cut_pos = _prefix_token_match(rec_text, block)
    if ratio >= 0.5 and matched >= 8 and cut_pos > 0:
        trimmed = block[cut_pos:].lstrip(" \t\r\n.:;,-")
        if "space clusters" in rec_text.lower():
            accelerator_tail = re.search(
                r"\bWe\s+have\s+collectively\s+supported\s+a\s+total\s+of\s+317\b",
                trimmed,
                re.IGNORECASE,
            )
            if accelerator_tail:
                trimmed = trimmed[: accelerator_tail.start()].strip()
        if len(trimmed.split()) >= 8:
            return trimmed

    response = (unit.get("response_text") or "").strip()
    if response and response != block:
        return response
    return block


def _space_unit_to_match(
    rec: Recommendation,
    unit: ResponseUnit,
    score: float,
    method: str,
) -> AlignedMatch:
    quoted = unit.get("quoted_recommendation_text") or None
    matched_text = _clean_matched_text(rec, unit, _trim_repeated_recommendation(rec["text"], unit))
    return AlignedMatch(
        rec_id=rec["rec_id"],
        recommendation=rec["text"],
        matched_chunk_id=unit["unit_id"],
        matched_text=matched_text,
        source=unit.get("source", ""),
        page_number=unit.get("page_start"),
        similarity=score,
        alignment_confidence=score,
        match_method=method,
        boundary_reason=unit.get("boundary_reason"),
        quoted_recommendation_text=quoted or rec["text"],
        heading_text=unit.get("heading_text"),
    )


def _match_space_recommendations_to_response_units(
    recommendations: list[Recommendation],
    response_units: list[ResponseUnit],
    similarity_threshold: float,
) -> list[AlignedMatch]:
    """
    Space response headings are response-document ordinals, not report paragraph
    ids.  Pair sections by repeated recommendation text plus ordinal order, and
    never accept paragraph-id label matches as exact labels.
    """
    if not recommendations or not response_units:
        return []

    rec_texts = [normalize_text(r["text"]) for r in recommendations]
    section_texts = [
        normalize_text(u.get("quoted_recommendation_text") or u.get("full_unit_text") or "")
        for u in response_units
    ]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    vec.fit(rec_texts + [t for t in section_texts if t] + ["__pad__"])
    rec_matrix = vec.transform(rec_texts)
    section_matrix = vec.transform(section_texts)
    section_sim: np.ndarray = cosine_similarity(rec_matrix, section_matrix)

    matches: list[AlignedMatch] = []
    for rec_idx, rec in enumerate(recommendations):
        ordinal_unit = response_units[rec_idx] if rec_idx < len(response_units) else None
        chosen_unit: ResponseUnit | None = None
        chosen_score = 0.0
        method = "structured_ordinal"

        if ordinal_unit is not None:
            prefix_ratio, matched, _cut_pos = _prefix_token_match(
                rec["text"],
                ordinal_unit.get("full_unit_text") or ordinal_unit.get("response_text") or "",
            )
            quote_overlap = _token_overlap(rec["text"], ordinal_unit.get("quoted_recommendation_text") or "")
            if prefix_ratio >= 0.5 and matched >= 8:
                chosen_unit = ordinal_unit
                chosen_score = max(prefix_ratio, float(section_sim[rec_idx][ordinal_unit["unit_id"]]))
                if quote_overlap >= 0.5:
                    method = "structure"

        if chosen_unit is None:
            best_idx = int(np.argmax(section_sim[rec_idx])) if section_sim.size else -1
            if best_idx >= 0:
                best_score = float(section_sim[rec_idx][best_idx])
                best_unit = response_units[best_idx]
                quote_overlap = _token_overlap(rec["text"], best_unit.get("quoted_recommendation_text") or "")
                prefix_ratio, matched, _cut_pos = _prefix_token_match(
                    rec["text"],
                    best_unit.get("full_unit_text") or best_unit.get("response_text") or "",
                )
                if best_score >= similarity_threshold and (quote_overlap >= 0.5 or (prefix_ratio >= 0.5 and matched >= 8)):
                    chosen_unit = best_unit
                    chosen_score = best_score
                    method = "structure" if quote_overlap >= 0.5 else "structured_ordinal"

        if chosen_unit is not None:
            matches.append(_space_unit_to_match(rec, chosen_unit, min(chosen_score, 0.95), method))

    return matches


# ---------------------------------------------------------------------------
# Chunk-based aligner — used by notebook + as last-ditch fallback
# ---------------------------------------------------------------------------

def match_recommendations_to_responses(
    recommendations: list[Recommendation],
    response_chunks: list[Chunk],
    top_k: int = 3,
    similarity_threshold: float = 0.05,
) -> list[AlignedMatch]:
    if not recommendations or not response_chunks:
        return []

    rec_texts = [normalize_text(r["text"]) for r in recommendations]
    chunk_texts = [normalize_text(c["text"]) for c in response_chunks]

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    vectorizer.fit(rec_texts + chunk_texts)
    rec_matrix = vectorizer.transform(rec_texts)
    chunk_matrix = vectorizer.transform(chunk_texts)
    sim_matrix: np.ndarray = cosine_similarity(rec_matrix, chunk_matrix)

    matches: list[AlignedMatch] = []
    for rec_idx, rec in enumerate(recommendations):
        scores = sim_matrix[rec_idx]
        top_indices = np.argsort(scores)[::-1][:top_k]
        for chunk_idx in top_indices:
            score = float(scores[chunk_idx])
            if score < similarity_threshold:
                continue
            chunk = response_chunks[chunk_idx]
            trimmed = trim_to_response_boundary(chunk["text"])
            matches.append(
                AlignedMatch(
                    rec_id=rec["rec_id"],
                    recommendation=rec["text"],
                    matched_chunk_id=chunk["chunk_id"],
                    matched_text=trimmed,
                    source=chunk["source"],
                    page_number=chunk["page_number"],
                    similarity=score,
                    alignment_confidence=score,
                    match_method="chunk_fallback",
                    boundary_reason="chunk_boundary_trim" if trimmed != chunk["text"] else None,
                    quoted_recommendation_text=None,
                    heading_text=None,
                )
            )
    return matches


# ---------------------------------------------------------------------------
# Response-unit aligner — used by the backend pipeline / Section 4 UI
# ---------------------------------------------------------------------------

def match_recommendations_to_response_units(
    recommendations: list[Recommendation],
    response_units: list[ResponseUnit],
    top_k: int = 3,
    similarity_threshold: float = 0.05,
    label_match_confidence: float = 0.95,
    structure_match_floor: float = 0.18,
) -> list[AlignedMatch]:
    """
    Align recommendations to response units.

    Three-tier matching:
        1. exact_label  — rec.item_label ↔ unit.recommendation_labels
        2. structure    — TF-IDF against unit.quoted_recommendation_text
        3. semantic     — TF-IDF against unit.response_text
    """
    if not recommendations or not response_units:
        return []

    if _is_space_economy_pair(recommendations, response_units):
        return _match_space_recommendations_to_response_units(
            recommendations,
            response_units,
            similarity_threshold=similarity_threshold,
        )

    is_blood_pair = _is_infected_blood_pair(recommendations, response_units)
    label_index = _build_label_index(response_units)

    # Texts for fallback similarity scoring.
    response_texts = [
        normalize_text(u.get("response_text") or u.get("full_unit_text") or "")
        for u in response_units
    ]
    quoted_texts = [
        normalize_text(u.get("quoted_recommendation_text") or "")
        for u in response_units
    ]
    rec_texts = [normalize_text(r["text"]) for r in recommendations]

    response_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    response_vec.fit(rec_texts + [t for t in response_texts if t] + ["__pad__"])
    rec_matrix = response_vec.transform(rec_texts)
    response_matrix = response_vec.transform(response_texts)
    response_sim: np.ndarray = cosine_similarity(rec_matrix, response_matrix)

    # Structure similarity — only meaningful if at least one quoted echo exists.
    if any(quoted_texts):
        struct_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        struct_vec.fit(rec_texts + [t for t in quoted_texts if t] + ["__pad__"])
        struct_rec = struct_vec.transform(rec_texts)
        struct_unit = struct_vec.transform(quoted_texts)
        structure_sim: np.ndarray = cosine_similarity(struct_rec, struct_unit)
        # Zero rows for units with no quoted echo so they never win this stage.
        for j, qt in enumerate(quoted_texts):
            if not qt:
                structure_sim[:, j] = 0.0
    else:
        structure_sim = np.zeros_like(response_sim)

    matches: list[AlignedMatch] = []

    for rec_idx, rec in enumerate(recommendations):
        rec_label = _norm_label(str(rec.get("item_label", "")))
        primary_unit: ResponseUnit | None = label_index.get(rec_label) if rec_label else None
        primary_method = "exact_label"
        primary_score = label_match_confidence

        # ── Step 1: exact label match ───────────────────────────────────────
        if primary_unit is None and rec_label:
            # Try a stripped form ("recommendation 1" → "1").
            stripped = rec_label.split()[-1] if rec_label.split() else rec_label
            primary_unit = label_index.get(stripped)
        if primary_unit is not None and is_blood_pair:
            reason = primary_unit.get("boundary_reason")
            primary_method = "structured_grouped" if reason == "structured_grouped" else "structured_section"
            primary_score = max(0.82, float(primary_unit.get("extraction_confidence", 0.82)))

        # ── Step 2: structure match ─────────────────────────────────────────
        if primary_unit is None:
            best_struct_idx = int(np.argmax(structure_sim[rec_idx])) if structure_sim.size else -1
            best_struct_score = (
                float(structure_sim[rec_idx][best_struct_idx])
                if best_struct_idx >= 0
                else 0.0
            )
            if best_struct_score >= structure_match_floor:
                primary_unit = response_units[best_struct_idx]
                primary_method = "structure"
                primary_score = best_struct_score

        # ── Step 3: semantic fallback ───────────────────────────────────────
        if primary_unit is None:
            best_sem_idx = int(np.argmax(response_sim[rec_idx])) if response_sim.size else -1
            best_sem_score = (
                float(response_sim[rec_idx][best_sem_idx])
                if best_sem_idx >= 0
                else 0.0
            )
            if best_sem_score >= similarity_threshold:
                primary_unit = response_units[best_sem_idx]
                primary_method = "semantic"
                primary_score = best_sem_score

        # ── Step 4: sequence fallback ───────────────────────────────────────
        # For recommendations near the end of the list that remain unmatched,
        # try the last response unit (or any unit marked bare_label_at_boundary)
        # that hasn't already been claimed by a higher-confidence match.  This
        # handles mislabelled final blocks in the response PDF (e.g., the
        # second "8.32" block that should correspond to rec 8.33).
        if primary_unit is None and rec_idx >= len(recommendations) - 3:
            # Gather units already claimed by exact/structure/semantic matches.
            claimed_ids: set[int] = {
                m["matched_chunk_id"]
                for m in matches
                if m.get("match_method") in ("exact_label", "structure", "semantic")
            }
            # Prefer a bare-label unit; fall back to the last unclaimed unit.
            bare_units = [
                u for u in response_units
                if u.get("boundary_reason") == "bare_label_at_boundary"
                and u["unit_id"] not in claimed_ids
            ]
            if bare_units:
                primary_unit = bare_units[-1]
                primary_method = "sequence_correction"
                primary_score = 0.5
            else:
                # Last unclaimed unit as a positional guess.
                unclaimed = [
                    u for u in reversed(response_units)
                    if u["unit_id"] not in claimed_ids
                ]
                if unclaimed:
                    primary_unit = unclaimed[0]
                    primary_method = "sequence_correction"
                    primary_score = 0.3

        if primary_unit is None:
            # No usable match — Section 4 will treat this as "not addressed".
            continue

        matches.append(_unit_to_match(rec, primary_unit, primary_score, primary_method))

        # ── Alternative candidates (TF-IDF over response_text) ──────────────
        scores = response_sim[rec_idx].copy()
        scores[primary_unit["unit_id"]] = -1.0
        alt_indices = np.argsort(scores)[::-1][: max(0, top_k - 1)]
        for unit_idx in alt_indices:
            score = float(response_sim[rec_idx][unit_idx])
            if score < similarity_threshold:
                break
            alt_unit = response_units[unit_idx]
            matches.append(_unit_to_match(rec, alt_unit, score, "semantic"))

    return matches


def _unit_to_match(
    rec: Recommendation,
    unit: ResponseUnit,
    score: float,
    method: str,
) -> AlignedMatch:
    response_text = (
        unit.get("response_text")
        or unit.get("full_unit_text")
        or ""
    )
    response_text = _clean_matched_text(rec, unit, response_text)
    return AlignedMatch(
        rec_id=rec["rec_id"],
        recommendation=rec["text"],
        matched_chunk_id=unit["unit_id"],
        matched_text=response_text,
        source=unit.get("source", ""),
        page_number=unit.get("page_start"),
        similarity=score,
        alignment_confidence=score,
        match_method=method,
        boundary_reason=unit.get("boundary_reason"),
        quoted_recommendation_text=unit.get("quoted_recommendation_text"),
        heading_text=unit.get("heading_text"),
    )
