"""Pydantic response models — mirrors src/ TypedDicts as serialisable shapes."""
from __future__ import annotations
from pydantic import BaseModel, Field


class StageStatusModel(BaseModel):
    status: str
    elapsed_ms: int | None
    error: str | None


class PipelineStatusResponse(BaseModel):
    stages: dict[str, StageStatusModel]
    preset_id: str | None
    preset_statuses: dict[str, str] = Field(default_factory=dict)
    preset_summaries: dict[str, dict[str, int | str]] = Field(default_factory=dict)
    policy_chunks: int
    response_chunks: int
    recommendations: int
    alignments: int
    labels: int
    load_ocr: bool = False
    semantic_available: bool = False
    embeddings_built: list[str] = Field(default_factory=list)


class SearchResultModel(BaseModel):
    chunk_id: int
    text: str
    source: str
    page_number: int | None
    score: float
    pair_id: str | None = None
    matched_terms: list[str] = Field(default_factory=list)
    confidence: str | None = None
    heading: str | None = None
    context_before: str | None = None
    context_after: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResultModel]
    query: str
    retriever: str
    scope: str
    elapsed_ms: int
    chunks_searched: int
    top_score: float | None
    query_terms: int
    results_returned: int
    query_type: str | None = None
    alpha: float | None = None


class RecommendationModel(BaseModel):
    rec_id: int
    text: str
    extraction_method: str
    detector: str
    confidence: float
    document: str
    page_number: int | str | None
    ocr: bool
    item_label: str
    span_id: str
    extraction_source: str = "primary"
    source_document_role: str = "policy"
    extraction_note: str | None = None
    source_paragraph: str | None = None
    source_item_type: str | None = None


class AlignedMatchModel(BaseModel):
    rec_id: int
    recommendation: str
    matched_chunk_id: int
    matched_text: str
    source: str
    page_number: int | None
    similarity: float
    alignment_confidence: float | None = None
    label: str | None = None   # populated after classification
    match_method: str | None = None
    boundary_reason: str | None = None
    quoted_recommendation_text: str | None = None
    heading_text: str | None = None


class EvaluationResponse(BaseModel):
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    per_class: dict[str, dict[str, float]]
    confusion_matrix: list[list[int]]


class PresetModel(BaseModel):
    id: str
    label: str
    dataset_group: str = "coursework_given"
    group_label: str = "Coursework given documents"
    group_description: str = ""
    is_extra: bool = False


class Task2SummaryModel(BaseModel):
    recommendations: int
    alignments: int
    classified: int
    mean_extraction_confidence: float
    mean_alignment_confidence: float


class Task2MatchModel(BaseModel):
    matched_chunk_id: int | None = None
    matched_text: str | None = None
    source: str | None = None
    page_number: int | None = None
    similarity: float = 0.0
    alignment_confidence: float = 0.0
    label: str | None = None
    label_display: str | None = None
    no_match: bool = False
    match_method: str | None = None
    boundary_reason: str | None = None
    quoted_recommendation_text: str | None = None
    heading_text: str | None = None


class Task2RecommendationModel(BaseModel):
    rec_id: int
    item_label: str
    text: str
    document: str
    page_number: int | str | None
    detector: str
    extraction_method: str
    confidence: float
    ocr: bool
    span_id: str | None = None
    matches: list[Task2MatchModel] = Field(default_factory=list)
    best_match: Task2MatchModel | None = None
    best_label: str | None = None
    label_display: str | None = None
    best_similarity: float = 0.0
    alignment_confidence: float = 0.0
    classification_confidence: float | None = None
    classifier_method: str = "rule_based"
    classification_rationale: str | None = None
    extraction_source: str = "primary"
    source_document_role: str = "policy"
    extraction_note: str | None = None
    source_paragraph: str | None = None
    source_item_type: str | None = None


class Task2ResultsResponse(BaseModel):
    preset_id: str | None
    stages: dict[str, str]
    summary: Task2SummaryModel
    recommendations: list[Task2RecommendationModel]
    evaluation: EvaluationResponse | None = None
    evaluation_status: str
