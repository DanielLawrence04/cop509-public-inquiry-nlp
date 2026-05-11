"""
Post Office Horizon — classification validation.

Integration tests assert per-row labels after the stricter classifier (strong
vs soft accept split, hedge override, accept-of-completed-action signals).

User-specified expectations (from Task 2 final quality pass):
  - rows 1, 2, 3, 6, 8, 10, 11, 12, 16, 18, 19 must be accepted
  - row 13 must be rejected
  - rows 4, 14, 15, 17 must be partially_accepted
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.classification import classify_response

PO_RESPONSE_PDF = (
    Path(__file__).parent.parent / "data" / "raw" / "PostOfficeHorizon-IT-Inquiry-Response.pdf"
)


@pytest.fixture(scope="module")
def po_classified():
    if not PO_RESPONSE_PDF.exists():
        pytest.skip(f"PDF not found: {PO_RESPONSE_PDF}")
    from src.pdf_loader import extract_pages
    from src.response_units import extract_response_units

    pages = extract_pages(str(PO_RESPONSE_PDF))
    units = extract_response_units(pages)
    label_to_result: dict[str, str] = {}
    for unit in units:
        resp_text = (unit.get("response_text") or "").strip()
        result = classify_response(resp_text)
        for lbl in unit.get("recommendation_labels") or [unit.get("recommendation_label")]:
            if lbl:
                label_to_result[lbl] = result
    return label_to_result


def test_count(po_classified):
    assert len(po_classified) == 19
    assert set(po_classified.keys()) == {str(i) for i in range(1, 20)}


@pytest.mark.parametrize("label", ["1", "2", "3", "6", "7", "8", "10", "11", "12", "16", "18", "19"])
def test_accepted_rows(po_classified, label):
    assert po_classified[label] == "accepted", (
        f"PO row {label} should be accepted; got {po_classified[label]!r}"
    )


def test_row_13_rejected(po_classified):
    assert po_classified["13"] == "rejected"


@pytest.mark.parametrize("label", ["4", "14", "15", "17"])
def test_partial_rows(po_classified, label):
    assert po_classified[label] == "partially_accepted", (
        f"PO row {label} should be partially_accepted; got {po_classified[label]!r}"
    )
