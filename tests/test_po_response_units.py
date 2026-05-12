"""
Post Office Horizon IT Inquiry regression test for response-unit extraction.

Run with:
    python -m pytest tests/test_po_response_units.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path

PO_RESPONSE_PDF = Path(__file__).parent.parent / "data" / "raw" / "PostOfficeHorizon-IT-Inquiry-Response.pdf"

# ---------------------------------------------------------------------------
# Expected response_text starts (first ~60 chars, prefix match).
# ---------------------------------------------------------------------------
EXPECTED_STARTS: dict[str, str] = {
    "1":  "DBT accepts this recommendation",
    "2":  "DBT accepts the Inquiry",
    "3":  "DBT accepts this recommendation",
    "4":  "DBT broadly accepts this recommendation",
    "5":  "From 9 October",
    "6":  "Sir Gary",
    "7":  "DBT agrees that the appropriate",
    "8":  "This does not require any retrospective",
    "9":  "People who have accepted the Fixed Sum Offer",
    "10": "DBT has updated the existing HSSA guidance",
    "11": "The minister announced on 8 July",
    "12": "All claimants in the GLOS",
    "13": "DBT rejects this recommendation",
    "14": "Prior to the publication",
    "15": "DBT broadly accepts this recommendation",
    "16": "DBT confirms that the HCRS",
    "17": "DBT sees clear advantages",
    "18": "Some family members of postmasters",
    "19": "DBT, the Post Office and Fuj",
}

# Expected end-of-response substrings (appear somewhere in last 200 chars).
EXPECTED_ENDS: dict[str, str] = {
    "4":  "They support DBT",
    "10": "across the schemes",
    "13": "with the benefit of funded legal advice",
}

# Rec 19 must NOT contain any of these front-matter strings.
FRONTMATTER_MARKERS = [
    "open-government-licence",
    "Crown copyright",
    "ISBN",
    "gov.uk/doc/open-government",
]

# Phrases that must NOT appear at the start of any response (recommendation voice).
BAD_RESPONSE_STARTS = (
    "recommendation 1",
    "recommendation 2",
    "recommendation 3",
    "recommendation 4",
    "recommendation 5",
    "recommendation 6",
    "recommendation 7",
    "recommendation 8",
    "recommendation 9",
    "recommendation 10",
    "recommendation 11",
    "recommendation 12",
    "recommendation 13",
    "recommendation 14",
    "recommendation 15",
    "recommendation 16",
    "recommendation 17",
    "recommendation 18",
    "recommendation 19",
    "hm government, the department",  # starts with rec text for Rec 1
    "the minister or the department",  # rec text for Rec 2
)


@pytest.fixture(scope="module")
def po_units():
    if not PO_RESPONSE_PDF.exists():
        pytest.skip(f"PDF not found: {PO_RESPONSE_PDF}")
    from src.pdf_loader import extract_pages
    from src.response_units import extract_response_units
    pages = extract_pages(str(PO_RESPONSE_PDF))
    return extract_response_units(pages)


def _label_map(units):
    mapping: dict[str, object] = {}
    for u in units:
        for lbl in (u.get("recommendation_labels") or [u.get("recommendation_label")]):
            if lbl:
                mapping[lbl] = u
    return mapping


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_unit_count(po_units):
    """Exactly 19 response units for 19 Post Office recommendations."""
    assert len(po_units) == 19, (
        f"Expected 19 units, got {len(po_units)}. "
        f"Labels: {[u.get('recommendation_label') for u in po_units]}"
    )


def test_all_labels_present(po_units):
    """Labels 1–19 all present, no missing, no duplicates."""
    lmap = _label_map(po_units)
    missing = [str(n) for n in range(1, 20) if str(n) not in lmap]
    assert not missing, f"Missing labels: {missing}"


def test_no_duplicate_labels(po_units):
    """No label maps to more than one unit."""
    seen: dict[str, int] = {}
    for u in po_units:
        for lbl in (u.get("recommendation_labels") or [u.get("recommendation_label")]):
            if lbl:
                seen[lbl] = seen.get(lbl, 0) + 1
    dups = {k: v for k, v in seen.items() if v > 1}
    assert not dups, f"Duplicate labels: {dups}"


def test_response_starts(po_units):
    """Every response_text starts with the expected prefix."""
    lmap = _label_map(po_units)
    failures = []
    for label, expected in EXPECTED_STARTS.items():
        unit = lmap.get(label)
        if unit is None:
            failures.append(f"{label}: NOT FOUND")
            continue
        resp = (unit.get("response_text") or "").strip()
        if not resp.lower().startswith(expected.lower()):
            failures.append(f"{label}: expected '{expected}...' got '{resp[:80]}'")
    assert not failures, "Response-start mismatches:\n" + "\n".join(failures)


def test_response_ends(po_units):
    """Responses for Rec 4, 10, 13 are not truncated early."""
    lmap = _label_map(po_units)
    failures = []
    for label, expected_end in EXPECTED_ENDS.items():
        unit = lmap.get(label)
        if unit is None:
            failures.append(f"{label}: NOT FOUND")
            continue
        resp = (unit.get("response_text") or "").strip()
        if expected_end.lower() not in resp.lower():
            failures.append(f"{label}: expected end marker '{expected_end}' not found in '{resp[-200:]}'")
    assert not failures, "Response-end truncation failures:\n" + "\n".join(failures)


def test_no_recommendation_text_leak(po_units):
    """No response starts with the recommendation block (rec text before the actual response)."""
    lmap = _label_map(po_units)
    failures = []
    for unit in po_units:
        resp = (unit.get("response_text") or "").strip().lower()
        for bad in BAD_RESPONSE_STARTS:
            if resp.startswith(bad.lower()):
                lbl = unit.get("recommendation_labels")
                failures.append(f"{lbl}: response starts with recommendation text '{bad}'")
                break
    assert not failures, "Recommendation-text leaks:\n" + "\n".join(failures)


def test_rec19_no_frontmatter(po_units):
    """Rec 19 must not contain copyright/front-matter text."""
    lmap = _label_map(po_units)
    unit = lmap.get("19")
    assert unit is not None, "Rec 19 not found"
    resp = (unit.get("response_text") or "").lower()
    full = (unit.get("full_unit_text") or "").lower()
    for marker in FRONTMATTER_MARKERS:
        assert marker.lower() not in resp, f"Rec 19 response contains front-matter: '{marker}'"
        assert marker.lower() not in full, f"Rec 19 full_unit_text contains front-matter: '{marker}'"


def test_key_content_markers(po_units):
    """Specific content markers must appear in their respective responses."""
    lmap = _label_map(po_units)
    checks = [
        ("5",  "From 9 October"),
        ("6",  "Sir Gary"),
        ("9",  "People who have accepted the Fixed Sum Offer"),
        ("11", "The minister announced on 8 July"),
        ("15", "DBT broadly accepts this"),
        ("19", "DBT, the Post Office and Fuj"),
    ]
    failures = []
    for label, marker in checks:
        unit = lmap.get(label)
        if unit is None:
            failures.append(f"{label}: NOT FOUND")
            continue
        resp = (unit.get("response_text") or "").strip()
        if marker.lower() not in resp.lower():
            failures.append(f"{label}: marker '{marker}' not found in response '{resp[:150]}'")
    assert not failures, "Content-marker failures:\n" + "\n".join(failures)
