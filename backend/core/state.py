"""In-memory pipeline state — single source of truth shared across all routes."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

try:
    import sentence_transformers as _st  # noqa: F401
    SEMANTIC_AVAILABLE: bool = True
except ImportError:
    SEMANTIC_AVAILABLE: bool = False
import copy

from src.chunking import Chunk
from src.extraction import Recommendation
from src.alignment import AlignedMatch
from src.classification import Label
from src.evaluation import EvaluationResult
from src.response_units import ResponseUnit

StageStatus = Literal["idle", "running", "done", "error"]


@dataclass
class StageState:
    status: StageStatus = "idle"
    elapsed_ms: int | None = None
    error: str | None = None


@dataclass
class PipelineSnapshot:
    stages: dict[str, StageState]
    policy_path: Path | None
    response_path: Path | None
    load_ocr: bool
    policy_pages: list[Any]
    response_pages: list[Any]
    policy_chunks: list[Chunk]
    response_chunks: list[Chunk]
    response_units: list[ResponseUnit]
    recommendations: list[Recommendation]
    alignments: list[AlignedMatch]
    labels: list[Label]
    evaluation: EvaluationResult | None
    evaluation_status: str | None


@dataclass
class PipelineState:
    # --- Stage statuses ---
    stages: dict[str, StageState] = field(default_factory=lambda: {
        "load":     StageState(),
        "extract":  StageState(),
        "align":    StageState(),
        "classify": StageState(),
    })

    # --- Active document pair ---
    preset_id: str | None = None
    policy_path: Path | None = None
    response_path: Path | None = None

    # --- Load stage outputs ---
    load_ocr: bool = False
    policy_pages: list[Any] = field(default_factory=list)
    response_pages: list[Any] = field(default_factory=list)
    policy_chunks:   list[Chunk] = field(default_factory=list)
    response_chunks: list[Chunk] = field(default_factory=list)
    response_units:  list[ResponseUnit] = field(default_factory=list)

    # --- Semantic embeddings cache keyed by preset_id ---
    embeddings_cache: dict[str, Any] = field(default_factory=dict)

    # --- Extraction stage outputs ---
    recommendations: list[Recommendation] = field(default_factory=list)

    # --- Alignment stage outputs ---
    alignments: list[AlignedMatch] = field(default_factory=list)

    # --- Classification stage outputs ---
    labels: list[Label] = field(default_factory=list)

    # --- Evaluation outputs ---
    evaluation: EvaluationResult | None = None
    evaluation_status: str | None = None

    # --- Cached outputs by preset ID ---
    preset_cache: dict[str, PipelineSnapshot] = field(default_factory=dict)

    def reset(self, clear_cache: bool = True) -> None:
        """Clear all pipeline outputs and reset stage statuses."""
        self.preset_id = None
        self.policy_path = None
        self.response_path = None
        self.load_ocr = False
        self.policy_pages.clear()
        self.response_pages.clear()
        self.policy_chunks.clear()
        self.response_chunks.clear()
        self.response_units.clear()
        self.recommendations.clear()
        self.alignments.clear()
        self.labels.clear()
        self.evaluation = None
        self.evaluation_status = None
        if clear_cache:
            self.preset_cache.clear()
            self.embeddings_cache.clear()
        for s in self.stages.values():
            s.status = "idle"
            s.elapsed_ms = None
            s.error = None

    def save_active(self) -> None:
        """Cache the currently active preset outputs, if a preset is active."""
        if self.preset_id is None:
            return

        self.preset_cache[self.preset_id] = PipelineSnapshot(
            stages={name: copy.copy(stage) for name, stage in self.stages.items()},
            policy_path=self.policy_path,
            response_path=self.response_path,
            load_ocr=self.load_ocr,
            policy_pages=list(self.policy_pages),
            response_pages=list(self.response_pages),
            policy_chunks=list(self.policy_chunks),
            response_chunks=list(self.response_chunks),
            response_units=list(self.response_units),
            recommendations=list(self.recommendations),
            alignments=list(self.alignments),
            labels=list(self.labels),
            evaluation=copy.deepcopy(self.evaluation),
            evaluation_status=self.evaluation_status,
        )

    def activate_cached(self, preset_id: str) -> bool:
        """Switch active state to a cached preset. Return False if not cached."""
        snapshot = self.preset_cache.get(preset_id)
        if snapshot is None:
            return False

        self.preset_id = preset_id
        self.policy_path = snapshot.policy_path
        self.response_path = snapshot.response_path
        self.load_ocr = snapshot.load_ocr
        self.policy_pages = list(snapshot.policy_pages)
        self.response_pages = list(snapshot.response_pages)
        self.stages = {name: copy.copy(stage) for name, stage in snapshot.stages.items()}
        self.policy_chunks = list(snapshot.policy_chunks)
        self.response_chunks = list(snapshot.response_chunks)
        self.response_units = list(snapshot.response_units)
        self.recommendations = list(snapshot.recommendations)
        self.alignments = list(snapshot.alignments)
        self.labels = list(snapshot.labels)
        self.evaluation = copy.deepcopy(snapshot.evaluation)
        self.evaluation_status = snapshot.evaluation_status

        # Self-heal stale snapshots: re-extract response units if missing,
        # OR if the existing units may have been built with an older extractor
        # (detected by checking whether any multi-label group has >= 2 labels
        # when the stored alignments suggest chunk_fallback was used).
        needs_reextract = self.response_pages and not self.response_units

        if not needs_reextract and self.response_pages and self.response_units and self.alignments:
            # If all alignments are chunk_fallback, the unit extractor probably
            # wasn't used — force re-extraction with the current code.
            all_methods = [a.get("match_method") for a in self.alignments]
            if all(m == "chunk_fallback" for m in all_methods if m):
                needs_reextract = True

        if needs_reextract:
            from src.response_units import extract_response_units
            try:
                fresh_units = extract_response_units(self.response_pages)
            except Exception:
                fresh_units = []
            if fresh_units:
                self.response_units = fresh_units
                # Force re-alignment + reclassification with the fresh units.
                self.alignments.clear()
                self.labels.clear()
                self.evaluation = None
                self.evaluation_status = None
                for stage_name in ("align", "classify"):
                    self.stages[stage_name].status = "idle"
                    self.stages[stage_name].elapsed_ms = None
                    self.stages[stage_name].error = None
                # Persist the refreshed snapshot so subsequent activates skip
                # re-extraction.
                self.save_active()
        return True

    def preset_statuses(self) -> dict[str, str]:
        """Return coarse UI status for cached presets."""
        statuses: dict[str, str] = {}
        for preset_id, snapshot in self.preset_cache.items():
            if any(stage.status == "error" for stage in snapshot.stages.values()):
                statuses[preset_id] = "error"
            elif snapshot.stages["classify"].status == "done":
                statuses[preset_id] = "complete"
            elif snapshot.stages["load"].status == "done":
                statuses[preset_id] = "loaded"
            else:
                statuses[preset_id] = snapshot.stages["load"].status
        return statuses

    def preset_summaries(self) -> dict[str, dict[str, int | str]]:
        """Return compact cached summary data for each preset."""
        statuses = self.preset_statuses()
        return {
            preset_id: {
                "status": statuses.get(preset_id, "idle"),
                "policy_chunks": len(snapshot.policy_chunks),
                "response_chunks": len(snapshot.response_chunks),
                "recommendations": len(snapshot.recommendations),
                "alignments": len(snapshot.alignments),
                "labels": len(snapshot.labels),
            }
            for preset_id, snapshot in self.preset_cache.items()
        }

    def stage_unlocked(self, stage: str) -> bool:
        """Return True if the stage's prerequisite is satisfied."""
        prereqs = {
            "load":     None,
            "extract":  "load",
            "align":    "extract",
            "classify": "align",
        }
        prereq = prereqs.get(stage)
        if prereq is None:
            return True
        return self.stages[prereq].status == "done"


# Module-level singleton shared across all routes
pipeline = PipelineState()
