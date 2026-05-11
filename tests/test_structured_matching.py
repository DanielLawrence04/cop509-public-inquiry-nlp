"""
Cross-document validation that response-unit extraction produces structured
sections (instead of falling back to chunk_fallback) for every document that
contains explicit "Recommendation N" / "Recommendation:" headings.

Each test is skipped when its source PDF is not present, so the suite is safe
to run anywhere.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.classification import classify_response
from src.response_units import extract_response_units

DATA_DIR = Path(__file__).parent.parent / "data" / "raw"

DOCS = {
    "covid_1":     DATA_DIR / "UK-Covid-19_Inquiry_Module_1_Response.pdf",
    "covid_2":     DATA_DIR / "UK-Covid-19_Inquiry_Module_2_Response.pdf",
    "grenfell":    DATA_DIR / "Grenfell-Phase2-Response.pdf",
    "summer_2024": DATA_DIR / "Summer2024-Disorder-Response.pdf",
}


def _load(doc: str):
    path = DOCS[doc]
    if not path.exists():
        pytest.skip(f"PDF not found: {path}")
    from src.pdf_loader import extract_pages
    return extract_response_units(extract_pages(str(path)))


def _label_map(units):
    out = {}
    for u in units:
        for lbl in u.get("recommendation_labels") or [u.get("recommendation_label")]:
            if lbl:
                out[lbl] = u
    return out


# ── Covid 1 ─────────────────────────────────────────────────────────────────

def test_covid1_structured():
    units = _load("covid_1")
    assert len(units) >= 10, f"Covid 1 expected ≥10 structured units; got {len(units)}"
    labels = _label_map(units)
    expected = {str(i) for i in range(1, 11)}
    missing = expected - set(labels.keys())
    assert not missing, f"Covid 1 missing labels: {sorted(missing)}"


# ── Covid 2 ─────────────────────────────────────────────────────────────────

def test_covid2_structured():
    units = _load("covid_2")
    assert len(units) >= 19, f"Covid 2 expected ≥19 structured units; got {len(units)}"
    labels = _label_map(units)
    expected = {str(i) for i in range(1, 20)}
    missing = expected - set(labels.keys())
    assert not missing, f"Covid 2 missing labels: {sorted(missing)}"


@pytest.mark.parametrize("label", ["2", "12"])
def test_covid2_substantive_rows_not_not_addressed(label):
    """User-flagged rows that previously came back as not_addressed."""
    units = _load("covid_2")
    labels = _label_map(units)
    assert label in labels, f"Covid 2 label {label} missing"
    result = classify_response((labels[label].get("response_text") or "").strip())
    assert result != "not_addressed", (
        f"Covid 2 row {label} classified as not_addressed; got {result!r}"
    )


@pytest.mark.parametrize("label", ["1", "13"])
def test_covid2_not_for_uk_government_rows_not_addressed(label):
    units = _load("covid_2")
    labels = _label_map(units)
    assert label in labels, f"Covid 2 label {label} missing"
    result = classify_response((labels[label].get("response_text") or "").strip())
    assert result == "not_addressed"


# ── Grenfell Phase 2 ─────────────────────────────────────────────────────────

def test_grenfell_structured():
    units = _load("grenfell")
    # Grenfell government response covers 58 recommendations; structural
    # extraction should produce at least 50 units (some may be grouped).
    assert len(units) >= 50, f"Grenfell expected ≥50 structured units; got {len(units)}"


# ── Summer 2024 Disorder ─────────────────────────────────────────────────────

def test_summer2024_structured_count():
    units = _load("summer_2024")
    assert len(units) == 9, f"Summer 2024 expected exactly 9 units; got {len(units)}"


def test_summer2024_sequential_labels():
    units = _load("summer_2024")
    labels = [u.get("recommendation_label") for u in units]
    assert labels == [str(i) for i in range(1, 10)], (
        f"Summer 2024 labels must be 1..9 in order; got {labels}"
    )


def test_summer2024_response_text_non_empty():
    units = _load("summer_2024")
    for u in units:
        rt = (u.get("response_text") or "").strip()
        assert len(rt) > 30, (
            f"Summer 2024 unit {u.get('recommendation_label')} has empty/short "
            f"response_text: {rt!r}"
        )


def test_summer2024_no_leak_between_units():
    """Recommendation 1's CPS-media-protocol reply must not bleed into 2 et al."""
    units = _load("summer_2024")
    labels = _label_map(units)
    rec1 = (labels.get("1", {}).get("response_text") or "").lower()
    rec2 = (labels.get("2", {}).get("response_text") or "").lower()
    assert "cps" in rec1 or "media protocol" in rec1 or "protocol" in rec1, (
        f"Summer 2024 rec 1 response missing CPS/protocol content: {rec1[:200]}"
    )
    assert "cps" not in rec2[:200] or "hmicfrs" in rec2 or "policing" in rec2, (
        f"Summer 2024 rec 2 looks like rec 1's content: {rec2[:200]}"
    )


# ── Summer 2024 recommendation-text cleanup ─────────────────────────────────

@pytest.fixture(scope="module")
def summer_recs():
    rec_pdf = DATA_DIR / "Summer2024-Disorder-Recomm.pdf"
    if not rec_pdf.exists():
        pytest.skip(f"PDF not found: {rec_pdf}")
    from src.extraction import extract_recommendations
    from src.pdf_loader import extract_pages
    return extract_recommendations(
        extract_pages(str(rec_pdf)),
        select_committee_section=True,
    )


@pytest.mark.parametrize("trailing", [
    "Policing response to disorder",
    "National policing response",
    "Impact on police forces",
    "The political response",
])
def test_summer2024_no_trailing_section_headings(summer_recs, trailing):
    """Recommendation text must not end with the next topic-section heading."""
    for r in summer_recs:
        body = (r.get("text") or "").strip()
        assert not body.endswith(trailing), (
            f"Rec {r.get('item_label')} ends with leaked heading: ...{body[-120:]!r}"
        )


# ── Summer 2024 row 2/3 classification ──────────────────────────────────────

def test_summer2024_row_2_partial():
    """Row 2 ('will set out how it intends to ...') should be partial."""
    units = _load("summer_2024")
    labels = _label_map(units)
    from src.classification import classify_response
    rt = (labels["2"].get("response_text") or "").strip()
    assert classify_response(rt) == "partially_accepted"


def test_summer2024_row_3_partial():
    """Row 3 ('options are being considered, no decisions taken') should be partial."""
    units = _load("summer_2024")
    labels = _label_map(units)
    from src.classification import classify_response
    rt = (labels["3"].get("response_text") or "").strip()
    assert classify_response(rt) == "partially_accepted"
