"""Pydantic request bodies."""
from pydantic import BaseModel, Field


class LoadPresetRequest(BaseModel):
    preset_id: str


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    retriever: str = Field(default="hybrid", pattern="^(tfidf|semantic|hybrid)$")
    scope: str = Field(default="current", pattern="^(current|all)$")


class AlignRequest(BaseModel):
    top_k: int = Field(default=3, ge=1, le=10)
    similarity_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
