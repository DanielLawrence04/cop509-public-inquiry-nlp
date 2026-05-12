"""
Grenfell Phase 2 — classification & extraction validation.

After the refined classifier and the response_units fixes:
  * Rows 50/51/54/55 must be partial (not not_addressed).  Their responses
    use the "The government supports this recommendation made towards local
    authorities…" pattern.
  * Row 58 must be accepted.  Boundary fix prevents Chapter 2 leak;
    paragraph-initial detection now handles "(113.83) The government accepts".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.classification import classify_response

GR_RESPONSE_PDF = Path(__file__).parent.parent / "data" / "raw" / "Grenfell-Phase2-Response.pdf"


@pytest.fixture(scope="module")
def gr_classified():
    if not GR_RESPONSE_PDF.exists():
        pytest.skip(f"PDF not found: {GR_RESPONSE_PDF}")
    from src.pdf_loader import extract_pages
    from src.response_units import extract_response_units

    pages = extract_pages(str(GR_RESPONSE_PDF))
    units = extract_response_units(pages)
    out: dict[str, str] = {}
    for u in units:
        labels = u.get("recommendation_labels") or [u.get("recommendation_label")]
        rt = (u.get("response_text") or "").strip()
        label = classify_response(rt)
        for lbl in labels:
            if lbl:
                out[lbl] = label
    return out


@pytest.fixture(scope="module")
def gr_units():
    if not GR_RESPONSE_PDF.exists():
        pytest.skip(f"PDF not found: {GR_RESPONSE_PDF}")
    from src.pdf_loader import extract_pages
    from src.response_units import extract_response_units

    pages = extract_pages(str(GR_RESPONSE_PDF))
    return {
        (u.get("recommendation_labels") or [u.get("recommendation_label")])[0]: u
        for u in extract_response_units(pages)
    }


@pytest.mark.parametrize("label", ["50", "51", "54", "55"])
def test_supports_recommendation_rows_are_partial(gr_classified, label):
    assert gr_classified.get(label) == "partially_accepted", (
        f"Grenfell {label} should be partial ('supports this recommendation'); got "
        f"{gr_classified.get(label)!r}"
    )


def test_row_58_accepted(gr_classified):
    assert gr_classified.get("58") == "accepted", (
        f"Grenfell 58 should be accepted; got {gr_classified.get('58')!r}"
    )


def test_row_58_chapter_leak_stripped(gr_units):
    """Row 58's response_text must not contain the Chapter 2 leak."""
    u = gr_units.get("58")
    assert u is not None, "Row 58 unit missing"
    rt = (u.get("response_text") or "")
    assert "Chapter 2" not in rt, f"Row 58 response_text leaks Chapter 2: {rt[:200]!r}"
    assert "Action taken since 2017" not in rt, (
        f"Row 58 response_text leaks 'Action taken since 2017': {rt[:200]!r}"
    )
