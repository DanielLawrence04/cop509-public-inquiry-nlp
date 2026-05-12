"""
Rule-based classification of government responses to recommendations.

Labels
------
accepted           – clear full acceptance + commitment to the specific action
rejected           – government explicitly declines the core ask
partially_accepted – agrees in principle, broadly accepts, hedges, defers to a
                     consultation/review, accepts only part, gives alternative
                     action, rejects the specific ask but offers related work,
                     or supports a recommendation made to a third party
not_addressed      – no substantive engagement with the recommendation

Priority order (highest to lowest)
----------------------------------
  1. Negated-accept / reject phrases
       + with partial/accept signal       → partially_accepted
       + alone                            → rejected
  2. Hedged-accept phrases ("broadly accepts", "accepts in principle",
     "accepts in part", "partially accept", "accept the thrust",
     "supports this recommendation") → partially_accepted
  3. Strong-accept signals (will implement / will deliver / has already
     published / committed to implementing / agrees that X should be Y /
     "will <action verb>" / completed-action verbs) → accepted, even when
     hedging signals coexist.
  4. Soft-accept signals (agree / accept / welcome / endorse) ALONE → accepted
  5. Soft-accept + STRONG hedge (will consider / will consult /
     "give careful consideration" / "feed into" / "commits to consult") →
     partially_accepted.  Generic engagement signals ("seeks to", "aims to",
     "working with", "will help") do NOT downgrade soft accept.
  6. Strong hedge or partial-engagement only → partially_accepted
  7. Nothing matched → not_addressed
"""

from __future__ import annotations

import re
from typing import Literal

from .utils import normalize_text  # noqa: F401 — kept for backward-compat imports

Label = Literal["accepted", "partially_accepted", "rejected", "not_addressed"]


# ── Negated-accept phrases ────────────────────────────────────────────────────
_NEGATED_ACCEPT = re.compile(
    r"\b(?:"
    r"does?\s+not\s+agree"
    r"|do\s+not\s+agree"
    r"|will\s+not\s+agree"
    r"|cannot\s+agree"
    r"|does?\s+not\s+accept"
    r"|do\s+not\s+accept"
    r"|will\s+not\s+accept"
    r"|does?\s+not\s+believe\s+(?:that\s+)?it\s+is\s+appropriate"
    r"|do\s+not\s+believe\s+(?:that\s+)?it\s+is\s+appropriate"
    r"|does?\s+not\s+consider\s+it\s+appropriate"
    r"|do\s+not\s+consider\s+it\s+appropriate"
    r"|does?\s+not\s+support"
    r"|do\s+not\s+support"
    r"|will\s+not\s+support"
    r"|cannot\s+support"
    r"|we\s+will\s+not\s+accept"
    r"|we\s+do\s+not\s+agree"
    r"|is\s+unable\s+to\s+accept"
    r"|is\s+not\s+able\s+to\s+accept"
    r")\b",
    re.IGNORECASE,
)

# ── Explicit reject patterns ──────────────────────────────────────────────────
_REJECT_PATTERNS = re.compile(
    r"\b(?:"
    r"reject[s]?"
    r"|disagree[sd]?"
    r"|will\s+not\b"
    r"|do\s+not\s+accept"
    r"|decline[sd]?"
    r"|cannot\s+accept"
    r"|not\s+accepted"
    r"|oppose[sd]?"
    r"|no\s+action"
    r"|refus(?:e[sd]?|ing)"
    r"|is\s+not\s+appropriate\s+to"
    r"|not\s+appropriate\s+to\s+(?:set|require|introduce|impose|mandate)"
    r"|recommendation\s+rejected"
    r"|recommendation\s+not\s+accepted"
    r")\b",
    re.IGNORECASE,
)

# ── Hedge-accept phrases ──────────────────────────────────────────────────────
# Acceptance with explicit qualifier — always partial.  Also covers
# "supports this recommendation" (Grenfell pattern: government endorses a
# recommendation made to another body without committing to action itself).
# Verbs use `\w*` suffix to catch accept/accepts/accepted, agree/agrees/agreed.
_HEDGE_ACCEPT = re.compile(
    r"\b(?:"
    r"broadly\s+(?:accept|agree|welcome|endorse|support)\w*"
    r"|(?:accept|agree|welcome|endorse)\w*\s+in\s+principle"
    r"|(?:accept|agree)\w*\s+in\s+part"
    r"|partly\s+(?:accept|agree)\w*"
    r"|partially\s+(?:accept|agree)\w*"
    r"|accept\w*\s+the\s+(?:thrust|spirit|intent|principle)"
    r"|agree\w*\s+with\s+the\s+(?:thrust|spirit|intent|principle)"
    r"|recommendation\s+(?:partially\s+accepted|partly\s+accepted|broadly\s+accepted|acknowledged|noted|accepted\s+in\s+principle)"
    # "In part" / "partially" as acceptance qualifier — even with concrete
    # action verbs in the same sentence ("will, in part, set out") this is
    # explicit partial acceptance.
    r"|in\s+part(?!\s+(?:to|of|by))"
    r"|partially\b(?!\s+accepted)"
    r"|recognise[sd]?\s+the\s+(?:importance|need|value|case)"
    r"|recognize[sd]?\s+the\s+(?:importance|need|value|case)"
    r"|acknowledge[sd]?\s+the\s+(?:importance|need|recommendation|concerns?)"
    r"|supports?\s+(?:this|the|that)\s+recommendation"
    r"|government\s+supports\s+(?:this|the|that)\s+recommendation"
    r"|accepting\s+in\s+principle"
    r"|accept\w*\s+(?:this|the)\s+recommendation\s+in\s+principle"
    r"|space\s+ministerial\s+forum"
    r")\b",
    re.IGNORECASE,
)

# ── Strong-accept TIER 1: concrete action / completed action ─────────────────
# Commitment to a specific concrete action or completed action.  Always means
# accepted, even when strong-hedge signals coexist (e.g. "will accredit
# providers" + "commits to consult on credible accreditation bodies").
#
# "make" deliberately omitted from the future-action verb list because "will
# make a statement" is itself a hedge (announcing intent to act later).
_STRONG_ACCEPT_TIER1 = re.compile(
    r"\b(?:"
    r"will\s+(?:implement|adopt|take\s+forward|bring\s+forward|deliver|"
    r"establish|introduce|publish|set\s+out|legislate|amend|create|appoint|"
    r"require|mandate|ensure|accredit|develop|build|expand|enact|enforce|"
    r"enable|fund|launch|reform|strengthen|undertake|update|upgrade|provide|"
    r"prepare|produce|prioritise|prioritize|complete|deploy|increase|improve|"
    r"modernise|modernize|change|pay|set\s+up)\w*"
    r"|already\s+(?:implemented|published|introduced|established|underway|adopted|amended|updated|appointed|delivered|completed|fulfilled|in\s+force)"
    r"|has\s+(?:already\s+)?(?:published|introduced|established|implemented|amended|adopted|updated|delivered|appointed|set\s+out|set\s+up|completed|fulfilled|enacted|legislated)"
    r"|have\s+(?:already\s+)?(?:published|introduced|established|implemented|amended|adopted|updated|delivered|appointed|set\s+out|set\s+up|completed|fulfilled|enacted|legislated|been\s+amended|been\s+updated|been\s+implemented|been\s+introduced|been\s+published|been\s+appointed|been\s+delivered)"
    r"|(?:has|have)\s+been\s+(?:appointed|amended|updated|published|implemented|established|completed|delivered|enacted|adopted)"
    r"|(?:has|have)\s+(?:jointly\s+)?embarked\s+on"
    r"|in\s+progress"
    r"|noted\s+and\s+accepted"
    r"|recommendation\s+accepted(?!\s+in\s+principle)"
    r"|recommendations?\s+[^.]{0,40}\s+are\s+accepted\s+in\s+full"
    r"|this\s+recommendation\s+is\s+accepted\s+in\s+full"
    r"|these\s+recommendations\s+are\s+accepted\s+in\s+full"
    r"|government\s+accepts?\s+(?:this|the|that)\s+recommendation(?!\s+in\s+principle)"
    r"|fulfilled\s+(?:it|this|the\s+recommendation)"
    r"|committed\s+to\s+(?:implement\w*|deliver\w*|introduc\w*|take\s+forward|bring\s+forward|set\s+up|set\s+out|improv\w*|strengthen\w*|enhanc\w*|legislat\w*|amend\w*|establish\w*|publish\w*|adopt\w*|appointing\b|appoint\w*)"
    r"|is\s+committed\s+to\s+(?:implement\w*|deliver\w*|introduc\w*|legislat\w*|amend\w*|publish\w*|establish\w*|adopt\w*|appoint\w*)"
    r"|(?:already\s+)?has\s+(?:a\s+|the\s+)?(?:remit|role|duty|responsibility)\s+(?:for|to)"
    r"|(?:already\s+)?have\s+(?:a\s+|the\s+)?(?:remit|role|duty|responsibility)\s+(?:for|to)"
    r")\b",
    re.IGNORECASE,
)

# ── Strong-accept TIER 2: directive endorsement ───────────────────────────────
# "Agrees that X should be Y/done" — endorses a specific directive claim about
# a concrete action.  Accepted UNLESS a strong-hedge phrase ("give careful
# consideration", "commits to consult on") also fires.  The action-verb list
# excludes generic "be" / "done" to avoid catching "more should be done"-style
# soft endorsements that coexist with explicit hedges of the core ask.
_STRONG_ACCEPT_TIER2 = re.compile(
    r"\b"
    r"agree\w*\s+that\s+(?:\w+\s+){1,8}should\s+(?:"
    r"have|provide|include|ensure|publish|carry|operate|require|adopt|"
    r"implement|deliver|amend|"
    r"be\s+(?:given|provided|published|implemented|introduced|adopted|"
    r"appointed|required|accredited|established|set|amended|legislated|"
    r"made\s+available|brought\s+forward)"
    r")\b",
    re.IGNORECASE,
)

# ── Soft-accept signals ───────────────────────────────────────────────────────
# Agreement words; stand-alone → accepted; with strong hedge → partial.
_SOFT_ACCEPT = re.compile(
    r"\b(?:"
    r"accept[s]?"
    r"|agree[sd]?"
    r"|endorse[sd]?"
    r"|welcome[sd]?"
    r")\b",
    re.IGNORECASE,
)

# ── Overriding hedge phrases ─────────────────────────────────────────────────
# Hard deferrals that override even tier-1 strong accept signals.  These are
# explicit statements that no decision has been made / a decision is pending,
# so any "will <action>" commitment in the same block is actually conditional.
_OVERRIDING_HEDGE = re.compile(
    r"\b(?:"
    r"no\s+decisions?\s+(?:have|has)\s+been\s+(?:taken|made)"
    r"|decisions?\s+(?:have|has)\s+not\s+been\s+(?:taken|made)"
    r"|are\s+being\s+considered"
    r"|will\s+set\s+out\s+how"
    r"|set\s+out\s+how\s+government\s+will\s+develop"
    r"|will\s+continue\s+to\s+be\s+considered"
    r"|continue\s+to\s+be\s+considered"
    r"|does\s+not\s+have\s+any\s+plans\s+to\s+rejoin"
    r"|will\s+consider\s+keeping"
    r"|being\s+considered\s+within\s+broader"
    r")\b",
    re.IGNORECASE,
)

# ── Strong hedge phrases ──────────────────────────────────────────────────────
# These DOWNGRADE soft accept to partial.  Limited to phrases that explicitly
# defer action: consultation about a decision, consideration, future work,
# partial commitment.  "will consult" is restricted to topic-consultation
# ("on/about/whether/how") so it does NOT match implementation consultation
# ("will consult [named parties] to ensure …").
_NOT_FOR_UK_GOVERNMENT = re.compile(
    r"\b(?:this\s+)?recommendation\s+is\s+not\s+for\s+"
    r"(?:the\s+)?UK\s+government\s+to\s+respond\s+to\b",
    re.IGNORECASE,
)

_LEAD_GOVERNMENT_MODEL_RETAINED = re.compile(
    r"\blead\s+government\s+department\s+model\b[^.]{0,140}"
    r"\bretain\w*\s+an\s+essential\s+role\b",
    re.IGNORECASE,
)

_STRONG_HEDGE = re.compile(
    r"\b(?:"
    r"will\s+consider"
    r"|will\s+consult\s+(?:on|about|with|whether|how)"
    r"|will\s+give\s+(?:careful|further|due)?\s*consideration"
    r"|give\s+(?:careful|further|due)\s+consideration"
    r"|commits?\s+(?:the\s+\w+(?:\s+\w+)?\s+)?to\s+(?:consult|consider|explore|review)"
    r"|will\s+feed\s+into"
    r"|feed\s+into\s+(?:the\s+)?(?:guidance|work|review|consultation|future)"
    r"|will\s+be\s+consulted"
    r"|will\s+be\s+setting\s+out"
    r"|set\s+out\s+how"
    r"|currently\s+reviewing"
    r"|in\s+the\s+process\s+of\s+reviewing"
    r"|intention\s+is\s+to\s+publish"
    r"|will\s+reflect\b"
    r"|will\s+explore"
    r"|will\s+look\s+at"
    r"|take\s+on\s+board"
    r"|further\s+consideration"
    r"|further\s+work"
    r"|under\s+consideration"
    r"|under\s+review"
    r"|to\s+consult\s+on\b"
    r"|plan[s]?\s+to"
    r"|intend[s]?\s+to"
    r"|noted\b(?!\s+and\s+accepted)"
    r"|are\s+being\s+considered"
    r"|no\s+decisions?\s+(?:have|has)\s+been\s+(?:taken|made)"
    r"|in\s+due\s+course"
    r"|in\s+the\s+future"
    r"|as\s+part\s+of\s+(?:the\s+)?(?:future|forthcoming)"
    # Statement-announcement hedge: "will make a (substantive) statement" is
    # itself a deferral — the action is announcing intent later.
    r"|will\s+make\s+a\s+(?:substantive\s+)?statement"
    # Alternative-action signal: "pursuing a strategy of [X different from rec]"
    r"|pursuing\s+a\s+(?:strategy|policy|approach)\s+of"
    # Contrastive partial acknowledgment.
    r"|we\s+do,?\s+however,?\s+recognise[sd]?\s+that"
    # Decision-pending hedges with required deliberation.
    r"|(?:will\s+need\s+to\s+be|need\s+to\s+be)\s+(?:carefully\s+|properly\s+|further\s+|due\s+)?(?:considered|explored|reviewed)"
    r"|are\s+(?:actively\s+)?considering\s+(?:its\s+)?options"
    # Existing-duty deflection: pointing to current statutory duties without
    # committing to action ("supports… but points to existing Housing Act
    # duties and guidance").
    r"|points?\s+to\s+(?:the\s+)?(?:existing|current)\s+(?:duties|guidance|requirements|provisions|act)"
    r"|refers?\s+to\s+(?:the\s+)?(?:existing|current)\s+(?:duties|guidance|requirements|provisions)"
    r")\b",
    re.IGNORECASE,
)

# ── Generic partial / engagement patterns ─────────────────────────────────────
# When alone (no soft accept), count as partial.  When co-occurring with soft
# accept, do NOT downgrade — these are descriptive engagement, not deferral.
_PARTIAL_PATTERNS = re.compile(
    r"\b(?:"
    r"aims?\s+to"
    r"|seek[s]?\s+to"
    r"|exploring"
    r"|review[s]?"
    r"|some\s+elements"
    r"|ongoing"
    r"|work\s+is\s+(?:also\s+)?underway"
    r"|is\s+underway"
    r"|committed\s+to\s+(?:review|consult|explore|consider|work|engaging|engage)"
    r"|will\s+work\s+(?:with|to\s+ensure|to\s+support|towards)"
    r"|working\s+with"
    r"|is\s+(?:developing|working|considering)"
    r"|are\s+(?:developing|working|considering)"
    r"|is\s+in\s+the\s+process\s+of"
    r"|are\s+in\s+the\s+process\s+of"
    r"|has\s+(?:been\s+developed|developed|worked)"
    r"|have\s+(?:been\s+developed|developed)"
    r"|has\s+been\s+maintained"
    r"|have\s+been\s+maintained"
    r"|has\s+invested"
    r"|have\s+invested"
    r"|invests?\s+in"
    r"|investment\s+in"
    r"|framework\s+agreement"
    r"|collaboration"
    r"|looks\s+forward\s+to\s+delivering"
    r"|already\s+(?:working|doing|considering|developing|looking)"
    r"|continues?\s+to"
    r"|continue[sd]?\s+to"
    r"|will\s+include"
    r"|will\s+be\s+included"
    r"|will\s+(?:also\s+)?help\b"
    r"|will\s+engage"
    r"|will\s+draw\s+on"
    r"|considering"
    r"|consider[s]?\s+(?:that|the|whether|how)"   # "considers that X" rather than bare "consider"
    r"|points?\s+to\s+(?:existing|current)"        # "points to existing duties"
    r"|refers?\s+to\s+(?:the\s+)?(?:existing|current)\s+(?:duties|guidance|requirements)"
    r"|committed\b"
    r")\b",
    re.IGNORECASE,
)


def normalize_label(label: str | None) -> Label:
    """Return the canonical Task 2 label for a possibly older label value."""
    value = str(label or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value in {"partial", "partially_accepted", "partially_accept"}:
        return "partially_accepted"
    if value == "accepted":
        return "accepted"
    if value == "rejected":
        return "rejected"
    return "not_addressed"


def classify_with_confidence(text: str) -> tuple[Label, float]:
    """
    Classify a response and return (label, confidence).

    Confidence is a heuristic 0.0–1.0 score reflecting how clearly the rule
    fired.  It is NOT a probability — the classifier is rule-based — but it
    lets the UI distinguish "many signals all agree" from "a single weak hit".
    """
    if not text or not text.strip():
        return "not_addressed", 0.0

    if _NOT_FOR_UK_GOVERNMENT.search(text):
        return "not_addressed", 0.9

    has_negated = bool(_NEGATED_ACCEPT.search(text))
    has_reject = bool(_REJECT_PATTERNS.search(text))
    has_hedge = bool(_HEDGE_ACCEPT.search(text))
    has_override = bool(_OVERRIDING_HEDGE.search(text))
    has_strong_t1 = bool(_STRONG_ACCEPT_TIER1.search(text))
    has_strong_t2 = bool(_STRONG_ACCEPT_TIER2.search(text))
    has_soft = bool(_SOFT_ACCEPT.search(text))
    has_strong_hedge = bool(_STRONG_HEDGE.search(text))
    has_partial = bool(_PARTIAL_PATTERNS.search(text))

    # 1. Rejection territory
    if has_negated or has_reject:
        if has_strong_t1 or has_strong_t2 or has_soft or has_partial or has_strong_hedge or has_hedge:
            return "partially_accepted", 0.7
        return "rejected", 0.85

    if _LEAD_GOVERNMENT_MODEL_RETAINED.search(text):
        return "partially_accepted", 0.75

    # 2. Hedged acceptance overrides any other accept signal
    if has_hedge:
        return "partially_accepted", 0.8

    # 2a. Hard deferral ("no decisions have been taken", "are being considered")
    # overrides even tier-1 strong commitments — any "will <action>" co-occurring
    # is conditional on the pending decision.
    if has_override:
        return "partially_accepted", 0.75

    # 3. Tier-1 strong: concrete action / completed action — accepted
    if has_strong_t1:
        return "accepted", 0.85 if (has_strong_hedge or has_partial) else 0.92

    # 4. Tier-2 strong: "agrees that X should be [specific action]" — accepted
    # only when no strong-hedge phrase is also present.
    if has_strong_t2:
        if has_strong_hedge:
            return "partially_accepted", 0.7
        return "accepted", 0.82

    # 5-6. Soft accept: downgrade only when a STRONG hedge phrase coexists.
    # Generic engagement (aims to, seeks to, will help, working with) does NOT
    # downgrade — those are descriptive, not deferrals.
    if has_soft:
        if has_strong_hedge:
            return "partially_accepted", 0.7
        return "accepted", 0.78 if has_partial else 0.75

    # 7. Strong hedge or partial-engagement only
    if has_strong_hedge:
        return "partially_accepted", 0.65
    if has_partial:
        return "partially_accepted", 0.6

    # 8. No signal
    return "not_addressed", 0.0


def classify_response(text: str) -> Label:
    """Classify and return only the label (backward-compat wrapper)."""
    label, _conf = classify_with_confidence(text)
    return label


def classify_batch(texts: list[str]) -> list[Label]:
    """Classify a list of response texts."""
    return [classify_response(t) for t in texts]
