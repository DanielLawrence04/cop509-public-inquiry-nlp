"""Pipeline stage endpoints."""
from __future__ import annotations
import time
import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from backend.core.state import pipeline, SEMANTIC_AVAILABLE
from backend.core.logger import log
from backend.core.presets import PRESETS, validate_preset_files
from backend.models.requests import LoadPresetRequest, AlignRequest
from backend.models.responses import (
    PipelineStatusResponse, StageStatusModel,
    RecommendationModel, AlignedMatchModel, EvaluationResponse, PresetModel,
    Task2MatchModel, Task2RecommendationModel, Task2ResultsResponse,
    Task2SummaryModel,
)

router = APIRouter()


# ── Presets ───────────────────────────────────────────────────────────────────

@router.get("/presets", response_model=list[PresetModel])
def list_presets():
    return [
        PresetModel(
            id=p.id,
            label=p.label,
            dataset_group=p.dataset_group,
            group_label=p.group_label,
            group_description=p.group_description,
            is_extra=p.is_extra,
        )
        for p in PRESETS.values()
    ]


# ── Status ────────────────────────────────────────────────────────────────────

@router.get("/debug/units")
def debug_units():
    """Return runtime proof of response-unit extraction for the active pair."""
    from src.response_units import extract_response_units, _INLINE_SPLIT
    import re

    full_text = "\n".join(p.get("text", "") for p in pipeline.response_pages)
    first_match_ctx = ""
    m = re.search(r'\brecommendation\b', full_text, re.IGNORECASE)
    if m:
        s = max(0, m.start() - 20)
        first_match_ctx = repr(full_text[s : s + 200])

    # Re-run extraction live on the active response_pages.
    fresh_units = extract_response_units(pipeline.response_pages)
    first_match_text = ""
    if pipeline.alignments:
        first_match_text = (pipeline.alignments[0].get("matched_text") or "")[:300]

    return {
        "preset_id": pipeline.preset_id,
        "response_pdf": str(pipeline.response_path),
        "response_pages_total": len(pipeline.response_pages),
        "response_pages_nonempty": sum(1 for p in pipeline.response_pages if (p.get("text") or "").strip()),
        "stored_response_units": len(pipeline.response_units),
        "fresh_response_units": len(fresh_units),
        "first_5_stored_labels": [u.get("recommendation_label") for u in pipeline.response_units[:5]],
        "first_5_fresh_labels": [u.get("recommendation_label") for u in fresh_units[:5]],
        "first_5_fresh_headings": [u.get("heading_text") for u in fresh_units[:5]],
        "align_path_used": "unit_aligner" if len(pipeline.response_units) >= 2 else "chunk_fallback",
        "fresh_align_path": "unit_aligner" if len(fresh_units) >= 2 else "chunk_fallback",
        "first_3_match_methods": [a.get("match_method") for a in pipeline.alignments[:3]],
        "first_match_matched_text_200": first_match_text,
        "first_recommendation_context_in_text": first_match_ctx,
        "full_text_length": len(full_text),
        "full_text_first_500": repr(full_text[:500]),
    }


@router.get("/debug/unit_detail")
def debug_unit_detail(label: str = "8.14"):
    """Dump full runtime data for a specific response unit label."""
    from src.response_units import extract_response_units, _build_label_index as _ru_build  # noqa
    from src.alignment import _build_label_index, _norm_label

    fresh_units = extract_response_units(pipeline.response_pages)
    label_idx = _build_label_index(fresh_units)
    norm = _norm_label(label)
    unit = label_idx.get(norm)

    # Also check 8.15 and 8.16 mapping
    related_labels = ["8.14", "8.15", "8.16"]
    related_mapping = {}
    for rl in related_labels:
        u = label_idx.get(_norm_label(rl))
        related_mapping[rl] = {
            "found": u is not None,
            "unit_id": u.get("unit_id") if u else None,
            "primary_label": u.get("recommendation_label") if u else None,
        }

    stale_alignments = []
    for a in pipeline.alignments:
        rec_label = str(a.get("recommendation", ""))[:30]
        stale_alignments.append({
            "rec_id": a.get("rec_id"),
            "match_method": a.get("match_method"),
            "matched_text_100": (a.get("matched_text") or "")[:100],
        })

    if unit is None:
        return {
            "error": f"Label {label!r} not found in fresh units",
            "fresh_unit_labels": [u.get("recommendation_label") for u in fresh_units],
            "related_mapping": related_mapping,
            "stale_alignments_count": len(pipeline.alignments),
        }

    return {
        "unit_id": unit.get("unit_id"),
        "recommendation_label": unit.get("recommendation_label"),
        "recommendation_labels": unit.get("recommendation_labels"),
        "boundary_reason": unit.get("boundary_reason"),
        "heading_text": unit.get("heading_text"),
        "extraction_confidence": unit.get("extraction_confidence"),
        "quoted_recommendation_text_500": (unit.get("quoted_recommendation_text") or "")[:500],
        "response_text_500": (unit.get("response_text") or "")[:500],
        "full_unit_text_1000": (unit.get("full_unit_text") or "")[:1000],
        "related_mapping": related_mapping,
        "stale_alignments_count": len(pipeline.alignments),
        "stale_alignment_sample": stale_alignments[:5],
        "fresh_units_total": len(fresh_units),
    }


@router.get("/status", response_model=PipelineStatusResponse)
def status():
    return PipelineStatusResponse(
        stages={
            name: StageStatusModel(
                status=s.status,
                elapsed_ms=s.elapsed_ms,
                error=s.error,
            )
            for name, s in pipeline.stages.items()
        },
        preset_id=pipeline.preset_id,
        preset_statuses=pipeline.preset_statuses(),
        preset_summaries=pipeline.preset_summaries(),
        policy_chunks=len(pipeline.policy_chunks),
        response_chunks=len(pipeline.response_chunks),
        recommendations=len(pipeline.recommendations),
        alignments=len(pipeline.alignments),
        labels=len(pipeline.labels),
        load_ocr=pipeline.load_ocr,
        semantic_available=SEMANTIC_AVAILABLE,
        embeddings_built=list(pipeline.embeddings_cache.keys()),
    )


# ── Reset ─────────────────────────────────────────────────────────────────────

@router.post("/reset")
def reset():
    pipeline.reset()
    log.emit("pipeline", "reset", "info")
    return {"ok": True}


@router.post("/activate/{preset_id}")
def activate_preset(preset_id: str):
    if preset_id not in PRESETS:
        raise HTTPException(status_code=404, detail=f"Unknown preset: {preset_id}")
    # Cold-backend auto-load: if the user picks any pair from the Search-tab
    # dropdown before the corpus has been built, populate the full 8-pair
    # search corpus on demand so activate works without requiring a
    # Documents-tab click. Covers both coursework_given and extra_found
    # presets - the local app treats all 8 as searchable.
    if preset_id not in pipeline.preset_cache:
        try:
            pairs, chunks = pipeline.ensure_default_search_corpus()
            if pairs:
                log.emit(
                    "pipeline.activate.autoload",
                    f"loaded {pairs} pairs, {chunks} chunks",
                    "ok",
                )
        except Exception as exc:
            log.emit("pipeline.activate.autoload", f"load failed: {exc}", "err")
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Search corpus could not be prepared "
                    f"({type(exc).__name__}: {exc})."
                ),
            )
    if not pipeline.activate_cached(preset_id):
        raise HTTPException(status_code=404, detail=f"No cached results for preset: {preset_id}")

    log.emit("pipeline.activate", f"({preset_id})", "ok")
    return {"ok": True}


# ── Stage 1: Load ─────────────────────────────────────────────────────────────

@router.post("/run/load")
async def run_load(body: LoadPresetRequest):
    if body.preset_id not in PRESETS:
        raise HTTPException(status_code=404, detail=f"Unknown preset: {body.preset_id}")

    preset = PRESETS[body.preset_id]
    try:
        validate_preset_files(preset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    pipeline.reset(clear_cache=False)
    pipeline.preset_id = preset.id
    pipeline.policy_path = preset.policy_pdf
    pipeline.response_path = preset.response_pdf
    pipeline.stages["load"].status = "running"

    await asyncio.to_thread(_load_sync, preset.policy_pdf, preset.response_pdf)
    return {"ok": True}


@router.post("/run/load/upload")
async def run_load_upload(policy: UploadFile = File(...), response: UploadFile = File(...)):
    """Accept two uploaded PDFs and run the load stage."""
    tmp = Path("outputs/cache")
    tmp.mkdir(parents=True, exist_ok=True)

    policy_path = tmp / policy.filename
    response_path = tmp / response.filename
    policy_path.write_bytes(await policy.read())
    response_path.write_bytes(await response.read())

    pipeline.reset(clear_cache=False)
    pipeline.preset_id = None
    pipeline.policy_path = policy_path
    pipeline.response_path = response_path
    pipeline.stages["load"].status = "running"

    await asyncio.to_thread(_load_sync, policy_path, response_path)
    return {"ok": True}


def _load_sync(policy_path: Path, response_path: Path) -> None:
    from src.pdf_loader import extract_pages
    from src.chunking import chunk_pages_v2

    try:
        t0 = time.monotonic()

        # Invalidate cached semantic embeddings for this preset so stale vectors
        # are not used if the documents are re-loaded with different content.
        if pipeline.preset_id in pipeline.embeddings_cache:
            del pipeline.embeddings_cache[pipeline.preset_id]

        log.emit("pdf_loader.extract_pages", f"({policy_path.name})", "info")
        policy_pages = extract_pages(policy_path)
        pipeline.policy_pages = list(policy_pages)
        if any(p["ocr"] for p in policy_pages):
            pipeline.load_ocr = True
        log.emit("pdf_loader.extract_pages", f"({policy_path.name})", "ok",
                 ms=int((time.monotonic() - t0) * 1000))

        t1 = time.monotonic()
        log.emit("pdf_loader.extract_pages", f"({response_path.name})", "info")
        response_pages = extract_pages(response_path)
        pipeline.response_pages = list(response_pages)
        if any(p["ocr"] for p in response_pages):
            pipeline.load_ocr = True
        log.emit("pdf_loader.extract_pages", f"({response_path.name})", "ok",
                 ms=int((time.monotonic() - t1) * 1000))

        t2 = time.monotonic()
        log.emit("chunking.chunk_pages_v2", f"(policy n={len(policy_pages)})", "info")
        pipeline.policy_chunks = chunk_pages_v2(policy_pages)
        log.emit("chunking.chunk_pages_v2", f"({len(pipeline.policy_chunks)} chunks)", "ok",
                 ms=int((time.monotonic() - t2) * 1000))

        t3 = time.monotonic()
        log.emit("chunking.chunk_pages_v2", f"(response n={len(response_pages)})", "info")
        pipeline.response_chunks = chunk_pages_v2(response_pages)
        log.emit("chunking.chunk_pages_v2", f"({len(pipeline.response_chunks)} chunks)", "ok",
                 ms=int((time.monotonic() - t3) * 1000))

        t4 = time.monotonic()
        from src.response_units import extract_response_units
        non_empty = sum(1 for p in response_pages if (p.get("text") or "").strip())
        joined_preview = " ".join(p.get("text", "") for p in response_pages)[:500]
        log.emit("response_units.extract_response_units",
                 f"(response n={len(response_pages)}, non_empty={non_empty}, "
                 f"text_preview={repr(joined_preview[:200])})", "info")
        pipeline.response_units = extract_response_units(response_pages)
        sample = ", ".join(
            f"[{','.join(u.get('recommendation_labels') or [u.get('recommendation_label') or '?'])}]"
            f"@p{u.get('page_start')}"
            for u in pipeline.response_units[:3]
        ) or "—"
        log.emit("response_units.extract_response_units",
                 f"({len(pipeline.response_units)} units; sample: {sample})", "ok",
                 ms=int((time.monotonic() - t4) * 1000))

        elapsed = int((time.monotonic() - t0) * 1000)
        pipeline.stages["load"].status = "done"
        pipeline.stages["load"].elapsed_ms = elapsed
        pipeline.save_active()
        log.emit("pipeline.load", "complete", "ok", ms=elapsed)

    except Exception as exc:
        pipeline.stages["load"].status = "error"
        pipeline.stages["load"].error = str(exc)
        pipeline.save_active()
        log.emit("pipeline.load", str(exc), "err")
        raise


# ── Stage 2: Extract ──────────────────────────────────────────────────────────

@router.post("/run/extract")
async def run_extract():
    if not pipeline.stage_unlocked("extract"):
        raise HTTPException(status_code=400, detail="Load stage must complete first")

    pipeline.recommendations.clear()
    pipeline.alignments.clear()
    pipeline.labels.clear()
    pipeline.evaluation = None
    pipeline.evaluation_status = None
    for stage_name in ("align", "classify"):
        pipeline.stages[stage_name].status = "idle"
        pipeline.stages[stage_name].elapsed_ms = None
        pipeline.stages[stage_name].error = None
    pipeline.stages["extract"].status = "running"
    await asyncio.to_thread(_extract_sync)
    return {"ok": True}


def _extract_sync() -> None:
    from src.extraction import extract_recommendations

    try:
        t0 = time.monotonic()
        log.emit("extraction.extract_recommendations",
                 f"(policy {len(pipeline.policy_pages)} pages)", "info")

        if not pipeline.policy_pages:
            raise ValueError("No policy page records are loaded; run the load stage first.")

        active_preset = PRESETS.get(pipeline.preset_id) if pipeline.preset_id else None
        response_fallback = (
            pipeline.response_pages
            if active_preset and active_preset.allow_response_heading_recommendation_fallback
            else None
        )
        inline_prefix = active_preset.inline_recommendation_chapter_prefix if active_preset else None
        select_committee = bool(active_preset and active_preset.select_committee_conclusions_section)
        pipeline.recommendations = extract_recommendations(
            pipeline.policy_pages,
            response_pages_fallback=response_fallback,
            inline_chapter_prefix=inline_prefix,
            select_committee_section=select_committee,
        )

        elapsed = int((time.monotonic() - t0) * 1000)
        pipeline.stages["extract"].status = "done"
        pipeline.stages["extract"].elapsed_ms = elapsed
        pipeline.save_active()
        log.emit("extraction.extract_recommendations",
                 f"({len(pipeline.recommendations)} recs)", "ok", ms=elapsed)

    except Exception as exc:
        pipeline.stages["extract"].status = "error"
        pipeline.stages["extract"].error = str(exc)
        pipeline.save_active()
        log.emit("extraction.extract_recommendations", str(exc), "err")
        raise


@router.get("/extract/results", response_model=list[RecommendationModel])
def extract_results():
    return [RecommendationModel(**r) for r in pipeline.recommendations]


# ── Stage 3: Align ────────────────────────────────────────────────────────────

@router.post("/run/align")
async def run_align(body: AlignRequest = AlignRequest()):
    if not pipeline.stage_unlocked("align"):
        raise HTTPException(status_code=400, detail="Extract stage must complete first")

    pipeline.alignments.clear()
    pipeline.labels.clear()
    pipeline.evaluation = None
    pipeline.evaluation_status = None
    pipeline.stages["classify"].status = "idle"
    pipeline.stages["classify"].elapsed_ms = None
    pipeline.stages["classify"].error = None
    pipeline.stages["align"].status = "running"
    await asyncio.to_thread(_align_sync, body.top_k, body.similarity_threshold)
    return {"ok": True}


def _align_sync(top_k: int, similarity_threshold: float) -> None:
    from src.alignment import (
        match_recommendations_to_responses,
        match_recommendations_to_response_units,
    )

    try:
        t0 = time.monotonic()

        # Defensive self-heal: if response_units is empty but we have response
        # pages, try extracting now.  This catches the case where a snapshot
        # was cached before the unit extractor was wired up.
        if not pipeline.response_units and pipeline.response_pages:
            from src.response_units import extract_response_units
            log.emit("alignment.self_heal",
                     f"(response_units empty; re-extracting from {len(pipeline.response_pages)} pages)",
                     "info")
            pipeline.response_units = extract_response_units(pipeline.response_pages)
            log.emit("alignment.self_heal",
                     f"({len(pipeline.response_units)} units after re-extract)", "ok")

        # Prefer response-unit alignment when units were successfully extracted.
        # A document with >= 2 units has meaningful structural splits; fewer
        # than 2 means unit extraction found no headings → fall back to chunks.
        use_units = len(pipeline.response_units) >= 2

        if use_units:
            log.emit("alignment.match_recommendations_to_response_units",
                     f"({len(pipeline.recommendations)} recs, "
                     f"{len(pipeline.response_units)} units, top_k={top_k})", "info")
            pipeline.alignments = match_recommendations_to_response_units(
                pipeline.recommendations,
                pipeline.response_units,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            log.emit("alignment.match_recommendations_to_response_units",
                     f"({len(pipeline.alignments)} matches)", "ok",
                     ms=int((time.monotonic() - t0) * 1000))
        else:
            log.emit("alignment.match_recommendations_to_responses",
                     f"({len(pipeline.recommendations)} recs, top_k={top_k}) [chunk fallback]", "info")
            pipeline.alignments = match_recommendations_to_responses(
                pipeline.recommendations,
                pipeline.response_chunks,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
            log.emit("alignment.match_recommendations_to_responses",
                     f"({len(pipeline.alignments)} matches)", "ok",
                     ms=int((time.monotonic() - t0) * 1000))

        elapsed = int((time.monotonic() - t0) * 1000)
        pipeline.stages["align"].status = "done"
        pipeline.stages["align"].elapsed_ms = elapsed
        pipeline.save_active()
        # Diagnostic: report the match_method of the first 3 alignments so we
        # can confirm Section 4 picked the unit aligner end-to-end.
        method_sample = ", ".join(
            f"rec{a.get('rec_id')}={a.get('match_method', '?')}"
            for a in pipeline.alignments[:3]
        ) or "—"
        log.emit(
            "alignment.align_sync",
            f"complete (path={'unit_aligner' if use_units else 'chunk_fallback'}, "
            f"units={len(pipeline.response_units)}, methods: {method_sample})",
            "ok",
            ms=elapsed,
        )

    except Exception as exc:
        pipeline.stages["align"].status = "error"
        pipeline.stages["align"].error = str(exc)
        pipeline.save_active()
        log.emit("alignment.match_recommendations_to_responses", str(exc), "err")
        raise


@router.get("/align/results", response_model=list[AlignedMatchModel])
def align_results():
    return [AlignedMatchModel(**a) for a in pipeline.alignments]


def _label_display(label: str | None) -> str | None:
    labels = {
        "accepted": "Accepted",
        "partially_accepted": "Partially accepted",
        "rejected": "Rejected",
        "not_addressed": "Not addressed",
    }
    return labels.get(label) if label is not None else None


def _task2_results_payload() -> Task2ResultsResponse:
    from src.classification import classify_response, classify_with_confidence, normalize_label

    stages = {name: stage.status for name, stage in pipeline.stages.items()}
    classify_done = pipeline.stages["classify"].status == "done"
    align_done = pipeline.stages["align"].status == "done"

    alignments_by_rec: dict[int, list[dict]] = {}
    for alignment in pipeline.alignments:
        rec_id = int(alignment["rec_id"])
        alignments_by_rec.setdefault(rec_id, []).append(dict(alignment))

    labels_by_rec: dict[int, str] = {}
    if classify_done:
        for rec, label in zip(pipeline.recommendations, pipeline.labels):
            labels_by_rec[int(rec["rec_id"])] = normalize_label(label)

    task2_recommendations: list[Task2RecommendationModel] = []
    for rec in pipeline.recommendations:
        rec_id = int(rec["rec_id"])
        rec_matches = sorted(
            alignments_by_rec.get(rec_id, []),
            key=lambda item: float(item.get("similarity", 0.0)),
            reverse=True,
        )

        match_models: list[Task2MatchModel] = []
        for match in rec_matches:
            match_label = None
            if classify_done:
                match_label = normalize_label(classify_response(str(match.get("matched_text", ""))))
            similarity = float(match.get("similarity", 0.0))
            match_models.append(
                Task2MatchModel(
                    matched_chunk_id=match.get("matched_chunk_id"),
                    matched_text=match.get("matched_text"),
                    source=match.get("source"),
                    page_number=match.get("page_number"),
                    similarity=similarity,
                    alignment_confidence=float(match.get("alignment_confidence", similarity)),
                    label=match_label,
                    label_display=_label_display(match_label),
                    no_match=False,
                    match_method=match.get("match_method"),
                    boundary_reason=match.get("boundary_reason"),
                    quoted_recommendation_text=match.get("quoted_recommendation_text"),
                    heading_text=match.get("heading_text"),
                )
            )

        best_match = match_models[0] if match_models else None
        best_similarity = best_match.similarity if best_match else 0.0
        best_label = labels_by_rec.get(rec_id)
        rationale = None
        classification_confidence: float | None = None

        if classify_done:
            if best_match is None:
                best_label = "not_addressed"
                classification_confidence = 0.0
                rationale = "No response match crossed the alignment threshold."
            else:
                # Recompute the rule-fire confidence from the matched text so
                # the UI can show how strong the classifier signal was.
                _lbl, classification_confidence = classify_with_confidence(
                    str(best_match.matched_text or "")
                )
                rationale = "Rule-based classifier applied to the highest-similarity response passage."
        elif align_done and best_match is None:
            best_label = "not_addressed"
            classification_confidence = 0.0
            rationale = "No response match crossed the alignment threshold."

        task2_recommendations.append(
            Task2RecommendationModel(
                rec_id=rec_id,
                item_label=str(rec.get("item_label", "")),
                text=str(rec["text"]),
                document=str(rec.get("document", "")),
                page_number=rec.get("page_number"),
                detector=str(rec.get("detector", "")),
                extraction_method=str(rec.get("extraction_method", "")),
                confidence=float(rec.get("confidence", 0.0)),
                ocr=bool(rec.get("ocr", False)),
                span_id=str(rec.get("span_id", "")) if rec.get("span_id") else None,
                matches=match_models,
                best_match=best_match,
                best_label=best_label,
                label_display=_label_display(best_label),
                best_similarity=best_similarity,
                alignment_confidence=best_match.alignment_confidence if best_match else 0.0,
                classification_confidence=classification_confidence,
                classifier_method="rule_based",
                classification_rationale=rationale,
                extraction_source=str(rec.get("extraction_source", "primary")),
                source_document_role=str(rec.get("source_document_role", "policy")),
                extraction_note=rec.get("extraction_note"),
                source_paragraph=rec.get("source_paragraph"),
                source_item_type=rec.get("source_item_type"),
            )
        )

    mean_extraction = 0.0
    if pipeline.recommendations:
        mean_extraction = (
            sum(float(r.get("confidence", 0.0)) for r in pipeline.recommendations)
            / len(pipeline.recommendations)
        )

    mean_alignment = 0.0
    if pipeline.alignments:
        mean_alignment = (
            sum(float(a.get("similarity", 0.0)) for a in pipeline.alignments)
            / len(pipeline.alignments)
        )

    evaluation_model = EvaluationResponse(**pipeline.evaluation) if pipeline.evaluation else None
    evaluation_status = pipeline.evaluation_status
    if evaluation_status is None:
        evaluation_status = "Classification has not run yet." if not classify_done else "Evaluation unavailable."

    return Task2ResultsResponse(
        preset_id=pipeline.preset_id,
        stages=stages,
        summary=Task2SummaryModel(
            recommendations=len(pipeline.recommendations),
            alignments=len(pipeline.alignments),
            classified=len(pipeline.labels) if classify_done else 0,
            mean_extraction_confidence=mean_extraction,
            mean_alignment_confidence=mean_alignment,
        ),
        recommendations=task2_recommendations,
        evaluation=evaluation_model,
        evaluation_status=evaluation_status,
    )


@router.get("/task2/results", response_model=Task2ResultsResponse)
def task2_results():
    return _task2_results_payload()


# ── Validated final-export passthrough ────────────────────────────────────────
# Read-only view of the validated coursework final export so the local UI can
# load the 246-row evidence file without re-running the full pipeline. Does
# not mutate pipeline state and never writes the JSON.

_FINAL_JSON_PATH = Path(__file__).resolve().parents[2] / "outputs" / "final_recommendations_246.json"

# Map the validated JSON's normalised classification strings to the
# backend's `best_label` vocabulary (so the frontend's existing LABEL_MAP
# adapter handles them unchanged).
_FINAL_LABEL_TO_BEST = {
    "accepted": "accepted",
    "partial": "partially_accepted",
    "partially_accepted": "partially_accepted",
    "rejected": "rejected",
    "not_addressed": "not_addressed",
}


def _final_row_to_task2_rec(idx: int, row: dict, preset) -> Task2RecommendationModel:
    debug = row.get("debug") or {}
    matched_text = row.get("matched_response_text")
    matched_page = row.get("matched_response_page")
    alignment_conf = float(debug.get("alignment_confidence", row.get("confidence", 0.0)) or 0.0)
    lex_sim = float(debug.get("lexical_similarity", alignment_conf) or 0.0)
    cls_conf = debug.get("classification_confidence")
    cls_conf = float(cls_conf) if cls_conf is not None else None
    best_label = _FINAL_LABEL_TO_BEST.get(str(row.get("classification") or ""), "not_addressed")

    best_match = None
    if matched_text:
        best_match = Task2MatchModel(
            matched_text=str(matched_text),
            source=str(preset.response_pdf.name) if preset else None,
            page_number=int(matched_page) if isinstance(matched_page, int) else (
                int(matched_page) if isinstance(matched_page, str) and matched_page.isdigit() else None
            ),
            similarity=lex_sim,
            alignment_confidence=alignment_conf,
            label=best_label,
            label_display=_label_display(best_label),
            match_method=str(debug.get("alignment_method") or "validated_final_export"),
        )

    rec_page = row.get("recommendation_page")
    page_number: int | str | None
    if isinstance(rec_page, int):
        page_number = rec_page
    elif isinstance(rec_page, str) and rec_page.strip():
        page_number = rec_page
    else:
        page_number = None

    return Task2RecommendationModel(
        rec_id=idx,
        item_label=str(row.get("id") or idx),
        text=str(row.get("recommendation_text") or ""),
        document=str(preset.policy_pdf.name) if preset else "",
        page_number=page_number,
        detector="validated_final_export",
        extraction_method="validated_final_export",
        confidence=float(row.get("confidence", 0.0) or 0.0),
        ocr=False,
        matches=[best_match] if best_match else [],
        best_match=best_match,
        best_label=best_label,
        label_display=_label_display(best_label),
        best_similarity=lex_sim,
        alignment_confidence=alignment_conf,
        classification_confidence=cls_conf,
        classifier_method=str(debug.get("classifier_method") or "rule_based"),
        classification_rationale="Loaded from validated coursework final export.",
        extraction_source="validated_final_export",
        source_document_role="policy",
    )


@router.get("/final-results")
def final_results():
    """Return the validated 246-row coursework export grouped by preset.

    Shape:
      {
        "source": "validated_final_export",
        "summary": {total, pair_counts, classification_distribution},
        "by_preset": {preset_id: Task2ResultsResponse-shaped dict},
      }
    """
    import json

    if not _FINAL_JSON_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Validated final export not found at {_FINAL_JSON_PATH}.",
        )

    with _FINAL_JSON_PATH.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    rows = payload.get("recommendations") or []

    rows_by_pair: dict[str, list[dict]] = {}
    classification_distribution: dict[str, int] = {}
    for row in rows:
        pair = str(row.get("document_pair") or "")
        rows_by_pair.setdefault(pair, []).append(row)
        cls = str(row.get("classification") or "not_addressed")
        classification_distribution[cls] = classification_distribution.get(cls, 0) + 1

    by_preset: dict[str, dict] = {}
    pair_counts: dict[str, int] = {}
    for preset_id, preset_rows in rows_by_pair.items():
        preset = PRESETS.get(preset_id)
        task2_recs = [
            _final_row_to_task2_rec(idx, row, preset)
            for idx, row in enumerate(preset_rows)
        ]
        mean_align = (
            sum(r.alignment_confidence for r in task2_recs) / len(task2_recs)
            if task2_recs else 0.0
        )
        mean_extract = (
            sum(r.confidence for r in task2_recs) / len(task2_recs)
            if task2_recs else 0.0
        )
        stages = {"load": "done", "extract": "done", "align": "done", "classify": "done"}
        response = Task2ResultsResponse(
            preset_id=preset_id,
            stages=stages,
            summary=Task2SummaryModel(
                recommendations=len(task2_recs),
                alignments=sum(1 for r in task2_recs if r.best_match is not None),
                classified=len(task2_recs),
                mean_extraction_confidence=mean_extract,
                mean_alignment_confidence=mean_align,
            ),
            recommendations=task2_recs,
            evaluation=None,
            evaluation_status=(
                "Validated final export: 246-row coursework evidence. "
                "Full manual ground-truth labels are unavailable, so accuracy/F1 "
                "metrics are not computed here — see Notebook 2 for the "
                "prediction-only evaluation."
            ),
        )
        by_preset[preset_id] = response.model_dump()
        pair_counts[preset_id] = len(task2_recs)

    return {
        "source": "validated_final_export",
        "exported_at": payload.get("exported_at"),
        "summary": {
            "total": len(rows),
            "pair_counts": pair_counts,
            "classification_distribution": classification_distribution,
        },
        "by_preset": by_preset,
    }


# ── Stage 4: Classify ─────────────────────────────────────────────────────────

@router.post("/run/classify")
async def run_classify():
    if not pipeline.stage_unlocked("classify"):
        raise HTTPException(status_code=400, detail="Align stage must complete first")

    pipeline.labels.clear()
    pipeline.evaluation = None
    pipeline.evaluation_status = None
    pipeline.stages["classify"].status = "running"
    await asyncio.to_thread(_classify_sync)
    return {"ok": True}


def _classify_sync() -> None:
    from src.classification import classify_response, normalize_label
    from src.evaluation import compare_to_ground_truth
    from src.utils import load_json

    try:
        t0 = time.monotonic()
        alignments_by_rec: dict[int, list[dict]] = {}
        for alignment in pipeline.alignments:
            alignments_by_rec.setdefault(int(alignment["rec_id"]), []).append(alignment)

        log.emit(
            "classification.classify_response",
            f"({len(pipeline.recommendations)} recommendations)",
            "info",
        )

        labels = []
        for rec in pipeline.recommendations:
            rec_matches = sorted(
                alignments_by_rec.get(int(rec["rec_id"]), []),
                key=lambda item: float(item.get("similarity", 0.0)),
                reverse=True,
            )
            if not rec_matches:
                labels.append("not_addressed")
                continue
            labels.append(normalize_label(classify_response(rec_matches[0]["matched_text"])))

        pipeline.labels = labels

        elapsed = int((time.monotonic() - t0) * 1000)
        pipeline.stages["classify"].status = "done"
        pipeline.stages["classify"].elapsed_ms = elapsed
        log.emit("classification.classify_response",
                 f"({len(pipeline.labels)} labels)", "ok", ms=elapsed)

        # Run evaluation only when the ground-truth schema matches the current
        # recommendation-level prediction set.
        try:
            gt_path = Path("data/ground_truth/labels.json")
            if not gt_path.exists():
                pipeline.evaluation = None
                pipeline.evaluation_status = "Ground truth file not found for the current prediction set."
            else:
                raw_ground_truth = load_json(gt_path)
                if isinstance(raw_ground_truth, dict) and isinstance(raw_ground_truth.get("labels"), list):
                    ground_truth = raw_ground_truth["labels"]
                elif isinstance(raw_ground_truth, list):
                    ground_truth = raw_ground_truth
                else:
                    ground_truth = None

                if ground_truth is None:
                    pipeline.evaluation = None
                    pipeline.evaluation_status = "Ground truth exists but is not a supported label list schema."
                elif len(ground_truth) != len(pipeline.labels):
                    pipeline.evaluation = None
                    pipeline.evaluation_status = (
                        "Ground truth label count does not match current recommendation-level "
                        f"predictions ({len(ground_truth)} ground truth vs {len(pipeline.labels)} predictions)."
                    )
                else:
                    pipeline.evaluation = compare_to_ground_truth(pipeline.labels, ground_truth)
                    pipeline.evaluation_status = "Evaluation available."
                    log.emit("evaluation.compare_to_ground_truth",
                             f"(acc={pipeline.evaluation['accuracy']:.3f})", "ok")
        except Exception as eval_exc:
            pipeline.evaluation = None
            pipeline.evaluation_status = f"Evaluation unavailable: {eval_exc}"
            log.emit("evaluation.compare_to_ground_truth", str(eval_exc), "err")

        pipeline.save_active()

    except Exception as exc:
        pipeline.stages["classify"].status = "error"
        pipeline.stages["classify"].error = str(exc)
        pipeline.save_active()
        log.emit("classification.classify_response", str(exc), "err")
        raise


@router.get("/classify/results")
def classify_results():
    results = []
    alignments_by_rec: dict[int, list[dict]] = {}
    for alignment in pipeline.alignments:
        alignments_by_rec.setdefault(int(alignment["rec_id"]), []).append(dict(alignment))

    for rec, label in zip(pipeline.recommendations, pipeline.labels):
        rec_matches = sorted(
            alignments_by_rec.get(int(rec["rec_id"]), []),
            key=lambda item: float(item.get("similarity", 0.0)),
            reverse=True,
        )
        best_match = rec_matches[0] if rec_matches else None
        results.append({
            **rec,
            "best_label": label,
            "label_display": _label_display(label),
            "best_match": best_match,
            "matches": rec_matches,
        })
    return results


@router.get("/classify/evaluation", response_model=EvaluationResponse)
def classify_evaluation():
    if pipeline.evaluation is None:
        raise HTTPException(status_code=404, detail="No evaluation results yet")
    return EvaluationResponse(**pipeline.evaluation)
