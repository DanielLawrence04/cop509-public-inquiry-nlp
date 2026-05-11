"""
Classification validation for Behaviour Change (8.1–8.33).

Covers two test levels:

1. Unit tests — classify_response() called directly with representative text
   fragments drawn from the known response starts (EXPECTED_STARTS in
   test_bc_response_units.py).  These run without any PDF or pipeline state.

2. Integration tests — extract response units from the PDF, then classify each
   unit's response_text and assert the corrected labels.  Skipped when the PDF
   is not present.

Known issues fixed by the updated classifier
--------------------------------------------
* "agrees" / "agreed" were not matched by the old \bagree\b pattern.
  All "The Government agrees…" responses were returning not_addressed.
  Fixed by using agree[sd]? in _ACCEPT_PATTERNS.

* "does not agree" was matched as ACCEPTED because the positive "agree"
  token fired before any reject check.  Fixed by _NEGATED_ACCEPT patterns
  which are evaluated before the accept patterns.

* "does not believe it is appropriate" had no reject rule.  Fixed by
  _NEGATED_ACCEPT.

* Partial patterns were too narrow; many commitment/review responses fell
  through to not_addressed.  Fixed by expanding _PARTIAL_PATTERNS.
"""
from __future__ import annotations

import pytest
from pathlib import Path

from src.classification import classify_response, Label

BC_RESPONSE_PDF = Path(__file__).parent.parent / "data" / "raw" / "Behaviour-Change-Response.pdf"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _check(text: str, expected: Label | tuple[Label, ...], *, note: str = "") -> None:
    """Assert classify_response(text) matches expected (scalar or set of labels)."""
    result = classify_response(text)
    allowed = expected if isinstance(expected, tuple) else (expected,)
    assert result in allowed, (
        f"classify_response({text!r}) → {result!r}; "
        f"expected one of {allowed}" + (f"  [{note}]" if note else "")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Level 1: Unit tests — no PDF required
# ─────────────────────────────────────────────────────────────────────────────

class TestAcceptPatterns:
    """agree[sd]? must match 'agree', 'agrees', 'agreed'."""

    def test_bare_agree(self):
        _check("We agree with the recommendation.", "accepted")

    def test_agrees(self):
        _check("The Government agrees with the Committee's conclusion.", "accepted",
               note="'agrees' must match; old \\bagree\\b boundary did not catch it")

    def test_agreed(self):
        _check("The Government agreed to implement the proposals.", "accepted")

    def test_government_agrees_prefix_81(self):
        _check("The Government agrees with the Committee's view on evidence-based policy.", "accepted",
               note="8.1 pattern")

    def test_government_agrees_prefix_85(self):
        _check("The Government agrees with the need to ensure departmental experts are consulted.", "accepted",
               note="8.5 pattern")

    def test_government_agrees_prefix_88(self):
        _check("The Government agrees that the best expertise in behavioural science should be linked to policy teams.", "accepted",
               note="8.8 pattern")

    def test_we_agree(self):
        _check("We agree that it is important for local authorities to tailor approaches.", "accepted",
               note="8.12 pattern")

    def test_welcomes(self):
        _check("The Government welcomes the Committee's recommendation.", "accepted")

    def test_already_published(self):
        _check("The Cabinet Office has already published guidance on this matter.", "accepted",
               note="8.10 pattern — 'already published' is an accept signal")

    def test_will_take_forward(self):
        _check("The Government will take forward the committee's recommendation.", "accepted")

    def test_is_committed_to(self):
        _check("The Government is committed to improving evaluation culture.", "accepted")


class TestRejectPatterns:
    """Negated-accept and explicit reject phrases must not be classified as accepted."""

    def test_does_not_agree(self):
        result = classify_response(
            "The Government does not agree with the Committee's view of the current "
            "Responsibility Deal pledges."
        )
        assert result != "accepted", (
            f"'does not agree' must not return accepted; got {result!r}  [8.25 pattern]"
        )

    def test_does_not_believe_appropriate(self):
        result = classify_response(
            "The Government does not believe that it is appropriate to set specific "
            "targets for reducing carbon emissions by reducing car use."
        )
        assert result in ("rejected", "partially_accepted"), (
            f"'does not believe it is appropriate' must not return accepted/not_addressed; got {result!r}  [8.33 pattern]"
        )
        assert result != "accepted", "8.33 must not be accepted"

    def test_will_not(self):
        _check("The Government will not introduce mandatory reporting requirements.",
               ("rejected", "partially_accepted"))

    def test_reject_keyword(self):
        _check("The Government rejects this recommendation.", "rejected")

    def test_disagree(self):
        _check("We disagree with the proposed approach.", "rejected")

    def test_does_not_support(self):
        result = classify_response("The Government does not support mandatory targets.")
        assert result != "accepted"

    def test_negated_accept_with_partial_gives_partial(self):
        """When a reject signal coexists with hedging language → partially_accepted."""
        result = classify_response(
            "The Government does not agree with the specific approach, "
            "but will consider further evidence before a final decision."
        )
        assert result == "partially_accepted", (
            f"reject + partial should give partially_accepted; got {result!r}"
        )


class TestPartialPatterns:
    """Commitment/review language must be caught as partial."""

    def test_will_consider(self):
        _check("The Government will consider this recommendation carefully.",
               "partially_accepted")

    def test_will_consult(self):
        _check("The Department will consult with stakeholders on the proposal.",
               "partially_accepted")

    def test_plans_to(self):
        _check("The Government plans to review the evidence base.", "partially_accepted")

    def test_intends_to(self):
        _check("The Government intends to work with local authorities on this issue.",
               "partially_accepted")

    def test_committed_to_review(self):
        _check("The Government is committed to reviewing the framework.", "partially_accepted")

    def test_will_work_with(self):
        _check("The Department will work with local councils to develop guidance.",
               "partially_accepted")

    def test_working_with(self):
        _check("The Government is working with industry to improve standards.",
               "partially_accepted")

    def test_continues_to(self):
        _check("The Government continues to support local authority efforts.",
               "partially_accepted")

    def test_aims_to(self):
        _check("The Government aims to strengthen the evidence base for local intervention.",
               "partially_accepted")

    def test_is_developing(self):
        _check("The Department is developing guidance for local commissioners.",
               "partially_accepted")

    def test_seeking_to(self):
        _check("The Government seeks to encourage a stronger evaluation culture.",
               "partially_accepted")

    def test_will_feed_into(self):
        _check("This work will feed into the broader programme review.",
               "partially_accepted")


class TestNotAddressed:
    """Generic or empty text must remain not_addressed."""

    def test_empty(self):
        _check("", "not_addressed")

    def test_whitespace(self):
        _check("   ", "not_addressed")

    def test_generic_context_statement(self):
        _check("The report was published in 2012.", "not_addressed")

    def test_no_stance(self):
        _check("Policy making should be evidence-based.", "not_addressed")


# ─────────────────────────────────────────────────────────────────────────────
# Level 2: Integration tests — require PDF + full pipeline extraction
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def bc_classified():
    """
    Returns dict mapping label → classify_response(response_text) for all
    Behaviour Change response units.  Skipped when PDF is absent.
    """
    if not BC_RESPONSE_PDF.exists():
        pytest.skip(f"PDF not found: {BC_RESPONSE_PDF}")
    from src.pdf_loader import extract_pages
    from src.response_units import extract_response_units

    pages = extract_pages(str(BC_RESPONSE_PDF))
    units = extract_response_units(pages)

    label_to_result: dict[str, str] = {}
    for unit in units:
        resp_text = (unit.get("response_text") or "").strip()
        result = classify_response(resp_text)
        for lbl in (unit.get("recommendation_labels") or [unit.get("recommendation_label")]):
            if lbl:
                label_to_result[lbl] = result
    return label_to_result


def test_total_rec_count(bc_classified):
    """All 33 labels 8.1–8.33 are present."""
    expected = {f"8.{i}" for i in range(1, 34)}
    missing = expected - set(bc_classified.keys())
    assert not missing, f"Missing classification results for: {sorted(missing)}"
    assert len(bc_classified) == 33, f"Expected 33, got {len(bc_classified)}"


@pytest.mark.parametrize("label", ["8.1", "8.2", "8.3", "8.4", "8.5", "8.8", "8.9"])
def test_government_agrees_not_not_addressed(bc_classified, label):
    """Responses that start 'The Government agrees…' must not be not_addressed."""
    result = bc_classified.get(label)
    assert result is not None, f"{label} missing from classified output"
    assert result != "not_addressed", (
        f"{label}: 'The Government agrees…' response classified as not_addressed; "
        f"got {result!r}"
    )


@pytest.mark.parametrize("label", ["8.20", "8.21", "8.22", "8.26", "8.30"])
def test_substantive_responses_not_not_addressed(bc_classified, label):
    """Responses with substantive government engagement must not be not_addressed."""
    result = bc_classified.get(label)
    assert result is not None, f"{label} missing from classified output"
    assert result != "not_addressed", (
        f"{label}: substantive response classified as not_addressed; got {result!r}"
    )


def test_8_25_not_accepted(bc_classified):
    """8.25 ('does not agree') must not be accepted."""
    result = bc_classified.get("8.25")
    assert result is not None, "8.25 missing from classified output"
    assert result != "accepted", (
        f"8.25 'does not agree' response should be rejected/partial, got {result!r}"
    )


def test_8_33_not_accepted(bc_classified):
    """8.33 ('does not believe it is appropriate') must not be accepted."""
    result = bc_classified.get("8.33")
    assert result is not None, "8.33 missing from classified output"
    assert result != "accepted", (
        f"8.33 'does not believe it is appropriate' should be rejected/partial, got {result!r}"
    )
    assert result in ("rejected", "partially_accepted"), (
        f"8.33 expected rejected or partially_accepted, got {result!r}"
    )


def test_classification_distribution(bc_classified):
    """Spot-check the overall distribution is not degenerate."""
    from collections import Counter
    counts = Counter(bc_classified.values())
    # After fixes, not_addressed should not dominate (was ~20+ before fix)
    assert counts.get("not_addressed", 0) < 15, (
        f"Too many not_addressed ({counts['not_addressed']}); classifier may be broken. "
        f"Full distribution: {dict(counts)}"
    )
    # There should be a mix of labels
    assert len(counts) >= 2, f"Only one label class found: {dict(counts)}"
    # Accepted should no longer dominate trivially — the stricter classifier
    # downgrades "agrees + hedge" to partial, so accepted < 15.
    assert counts.get("accepted", 0) < 20, (
        f"Too many accepted ({counts['accepted']}); classifier may be over-accepting. "
        f"Full distribution: {dict(counts)}"
    )


@pytest.mark.parametrize("label", ["8.6", "8.20", "8.21", "8.22", "8.27"])
def test_hedged_rows_are_partial(bc_classified, label):
    """Rows whose response hedges ('agrees... will consider/feed into/consult') must be partial, not accepted."""
    result = bc_classified.get(label)
    assert result is not None, f"{label} missing from classified output"
    assert result == "partially_accepted", (
        f"{label} should be partially_accepted (hedge after 'agrees'); got {result!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Two-tier strong-accept regression tests (refined classifier pass)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("label", ["8.1", "8.2", "8.5"])
def test_directly_relevant_agreement_is_accepted(bc_classified, label):
    """
    "The Government agrees with the Committee's conclusion" + directly
    relevant response (no will-consider/will-consult/feed-into hedge of the
    core ask) should be accepted, not partial.
    """
    result = bc_classified.get(label)
    assert result is not None, f"{label} missing"
    assert result == "accepted", (
        f"{label} should be accepted (direct agreement, no strong-hedge of core ask); got {result!r}"
    )


def test_in_part_qualifier_forces_partial():
    """'will, in part, set out' is partial acceptance, not full."""
    text = "The Police Reform White Paper will, in part, set out the future of policing."
    assert classify_response(text) == "partially_accepted"


def test_supports_recommendation_pattern_is_partial():
    """Grenfell pattern: 'The government supports this recommendation made towards local authorities…'"""
    text = (
        "The government supports this recommendation made towards local authorities. "
        "This duty will be highlighted in guidance that clarifies key responsibilities."
    )
    assert classify_response(text) == "partially_accepted"


def test_accepted_in_principle_is_partial():
    """'accepted in principle' is always partial, never fully accepted."""
    assert classify_response("The government accepts this recommendation in principle.") == "partially_accepted"
    assert classify_response("Recommendation accepted in principle.") == "partially_accepted"


def test_overriding_hedge_blocks_strong_t1():
    """'no decisions have been taken' + 'will require' = partial, not accepted."""
    text = (
        "We have considered the proposals which will require financial investment. "
        "Decisions have not been taken on the form that stronger capability may take "
        "and all options are being considered."
    )
    assert classify_response(text) == "partially_accepted"


def test_agrees_that_x_should_be_given_is_accepted():
    """Tier-2 strong: directive endorsement with concrete action verb."""
    text = (
        "DBT agrees that the appropriate powers should be given to the appointed person. "
        "DBT will consult Sir Ross Cranston to ensure the appointee has the relevant powers."
    )
    # 'will consult [Named Person]' is NOT a strong hedge; tier-2 strong fires.
    assert classify_response(text) == "accepted"


def test_no_duplicate_labels(bc_classified):
    """Each label 8.1–8.33 appears at most once."""
    from collections import Counter
    counts = Counter(bc_classified.keys())
    dupes = {k: v for k, v in counts.items() if v > 1}
    assert not dupes, f"Duplicate label entries: {dupes}"


def test_no_blank_labels(bc_classified):
    """No label key is empty or None."""
    for lbl, result in bc_classified.items():
        assert lbl and lbl.strip(), f"Blank label key found with result {result!r}"
