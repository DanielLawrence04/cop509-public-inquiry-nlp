"""PDF page preview — renders a page as PNG with optional text highlights."""
from __future__ import annotations
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from backend.core.state import pipeline

router = APIRouter()


@router.get("/")
def preview_page(
    doc: str = Query(..., description="'policy' or 'response'"),
    page: int = Query(..., ge=1),
    snippet: str = Query(default=""),
    dpi: int = Query(default=150, ge=72, le=300),
):
    if doc == "policy":
        pdf_path = pipeline.policy_path
    elif doc == "response":
        pdf_path = pipeline.response_path
    else:
        raise HTTPException(status_code=400, detail="doc must be 'policy' or 'response'")

    if pdf_path is None or not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Document not loaded")

    from src.preview import render_page_with_highlights
    result = render_page_with_highlights(pdf_path, page_number=page, snippet=snippet, dpi=dpi)

    return Response(
        content=result.png_bytes,
        media_type="image/png",
        headers={
            "X-Match-Count": str(result.match_count),
            "X-Match-Strategy": result.match_strategy,
        },
    )
