"""
Behaviour Change regression test for response-unit extraction.

Compares extracted response_text starts against manually verified expected
starts for all Behaviour Change recommendations (8.1 – 8.33).

Run with:
    python -m pytest tests/test_bc_response_units.py -v
"""
from __future__ import annotations

import pytest
from pathlib import Path

BC_RESPONSE_PDF = Path(__file__).parent.parent / "data" / "raw" / "Behaviour-Change-Response.pdf"

# ---------------------------------------------------------------------------
# Expected response_text starts (first ~60 chars, enough for a prefix match).
# Key is the CANONICAL primary label for the unit; shared-response units are
# checked via every label in the group.
# ---------------------------------------------------------------------------
EXPECTED_STARTS: dict[str, str] = {
    "8.1":  "The Government agrees with the Committee",
    "8.2":  "The Government agrees with the Committee",
    "8.3":  "The Government agrees with the Committee",   # 8.3/8.4 shared
    "8.5":  "The Government agrees with the need to ensure",
    "8.6":  "It is essential that policy making is informed",
    "8.8":  "The Government agrees that the best expertise",
    "8.10": "The Cabinet Office has already published",
    "8.12": "We agree that it is important",
    "8.14": "The Government agrees with the conclusion that",
    "8.17": "As the Committee has acknowledged",
    "8.20": "The Government agrees that it is important for local",
    "8.21": "The Government agrees with the Committee",
    "8.23": "The Government agrees with the Committee",
    "8.24": "Food labelling is an area of EU competence",
    "8.25": "The Government does not agree with the Committee",
    "8.26": "To help local areas successfully tackle",
    "8.27": "The Government is certainly anti-carbon",
    "8.28": "As stated above, it is not a question",
    "8.30": "Local policy packages, such as those funded",
    "8.31": "In launching the Local Sustainable Transport Fund",
    "8.32": "The Department for Transport drew local authorities",
    "8.33": "The Government does not believe that it is appropriate",
}

# Labels that should appear in the same unit as their primary (grouped responses)
EXPECTED_GROUPS: dict[str, list[str]] = {
    "8.3":  ["8.3", "8.4"],
    "8.6":  ["8.6", "8.7"],
    "8.8":  ["8.8", "8.9"],
    "8.10": ["8.10", "8.11"],
    "8.12": ["8.12", "8.13"],
    "8.14": ["8.14", "8.15", "8.16"],
    "8.17": ["8.17", "8.18", "8.19"],
    "8.21": ["8.21", "8.22"],
    "8.28": ["8.28", "8.29"],
}

# Recommendation-voice prefixes that must NOT appear at the start of a response
_BAD_RESPONSE_STARTS = (
    "we recommend",
    "we urge",
    "we invite",
    "we draw attention",
    "although we welcome",
    "whilst we welcome",
    "while we welcome",
    "we note, however",
)


@pytest.fixture(scope="module")
def bc_units():
    if not BC_RESPONSE_PDF.exists():
        pytest.skip(f"PDF not found: {BC_RESPONSE_PDF}")
    from src.pdf_loader import extract_pages
    from src.response_units import extract_response_units
    pages = extract_pages(str(BC_RESPONSE_PDF))
    return extract_response_units(pages)


def _label_map(units):
    """Build label → unit mapping covering all labels in each unit's group."""
    mapping: dict[str, object] = {}
    for u in units:
        for lbl in (u.get("recommendation_labels") or [u.get("recommendation_label")]):
            if lbl:
                mapping[lbl] = u
    return mapping


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_unit_count(bc_units):
    """Exactly 22 response units for 33 recommendations (11 grouped + 22 singletons)."""
    assert len(bc_units) == 22, (
        f"Expected 22 units, got {len(bc_units)}. "
        f"Labels: {[u.get('recommendation_label') for u in bc_units]}"
    )


def test_response_starts(bc_units):
    """Every expected response_text starts with the correct prefix."""
    lmap = _label_map(bc_units)
    failures = []
    for label, expected in EXPECTED_STARTS.items():
        unit = lmap.get(label)
        if unit is None:
            failures.append(f"{label}: NOT FOUND in extracted units")
            continue
        resp = (unit.get("response_text") or "").strip()
        prefix = expected[:55]
        if not resp.lower().startswith(prefix.lower()):
            failures.append(
                f"{label}: expected '{prefix}...' got '{resp[:80]}'"
            )
    assert not failures, "Response-start mismatches:\n" + "\n".join(failures)


def test_grouped_responses(bc_units):
    """Grouped recommendations map to the same response unit."""
    lmap = _label_map(bc_units)
    failures = []
    for primary, group in EXPECTED_GROUPS.items():
        units_in_group = {id(lmap[lbl]) for lbl in group if lbl in lmap}
        missing = [lbl for lbl in group if lbl not in lmap]
        if missing:
            failures.append(f"Group {group}: labels {missing} not found")
        elif len(units_in_group) > 1:
            failures.append(f"Group {group}: labels map to {len(units_in_group)} units (expected 1)")
    assert not failures, "Group mapping failures:\n" + "\n".join(failures)


def test_8_25_8_26_split(bc_units):
    """8.25 and 8.26 must be separate response units."""
    lmap = _label_map(bc_units)
    assert "8.25" in lmap, "8.25 not found"
    assert "8.26" in lmap, "8.26 not found"
    assert id(lmap["8.25"]) != id(lmap["8.26"]), "8.25 and 8.26 incorrectly merged"


def test_8_33_sequence_correction(bc_units):
    """The mislabelled final bare-label block is corrected to 8.33."""
    lmap = _label_map(bc_units)
    assert "8.33" in lmap, "8.33 not found after sequence correction"
    unit = lmap["8.33"]
    resp = (unit.get("response_text") or "").strip()
    assert resp.lower().startswith("the government does not believe"), (
        f"8.33 response unexpected start: '{resp[:80]}'"
    )


def test_no_recommendation_voice_leak(bc_units):
    """No response_text should begin with committee-voice recommendation language."""
    failures = []
    for u in bc_units:
        resp = (u.get("response_text") or "").strip().lower()
        for bad in _BAD_RESPONSE_STARTS:
            if resp.startswith(bad):
                failures.append(
                    f"{u.get('recommendation_labels')}: response starts with '{bad}'"
                )
                break
    assert not failures, "Recommendation-voice leaks in response_text:\n" + "\n".join(failures)


def test_all_labels_covered(bc_units):
    """All 33 Behaviour Change labels 8.1–8.33 have a matching response unit."""
    lmap = _label_map(bc_units)
    all_expected = (
        list(EXPECTED_STARTS.keys())
        + ["8.4", "8.7", "8.9", "8.11", "8.13", "8.15", "8.16", "8.18", "8.19", "8.22", "8.29"]
    )
    missing = [lbl for lbl in all_expected if lbl not in lmap]
    assert not missing, f"Labels missing from extracted units: {missing}"
