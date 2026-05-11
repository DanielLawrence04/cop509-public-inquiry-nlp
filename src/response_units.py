"""
Response-unit extraction for government response documents.

A *response unit* is the government's answer to a single recommendation
(or to a small group of recommendations sharing one block).

Critical preprocessing note
---------------------------
The text arriving in ``PageRecord["text"]`` has already been through
``_normalise_page_text`` in preprocessing.py which:
  1. Strips "Hol Inquiry" completely (line 426).
  2. Collapses every single newline to a space (line 453).

This means the page text is effectively one flat string per page.  The
heading "Hol Inquiry Recommendation\n\n8.1." becomes the inline token
"Recommendation 8.1." in the middle of a paragraph.  A regex that
requires ^|\\n before the keyword will NEVER fire on this text.

Strategy
--------
1. Scan the full joined text for every occurrence of
   "Recommendation N.M" (inline, no line-start requirement).
2. Cluster consecutive occurrences that belong to the same structural
   split point (same heading block).
3. For each cluster, extract the text up to the next cluster as the
   full unit block.
4. Split the unit block into (quoted_recommendation, response_text) by
   searching for a paragraph-initial government-response opener.
5. Filter out matches that have no response opener AND no substantial
   text (< MIN_BLOCK threshold) — those are inline references, not
   structural headings.

This correctly handles:
  * Behaviour Change "Recommendation 8.1. The idea..."
  * Space Economy "Recommendation 1 We acknowledge..."
  * Multi-label blocks "Recommendation 8.10. ... Recommendation 8.11."
  * Docs with properly preserved newlines (still scanned correctly)
"""
from __future__ import annotations

import re
from typing import TypedDict

from .pdf_loader import PageRecord


class ResponseUnit(TypedDict, total=False):
    unit_id: int
    source: str
    recommendation_label: str | None        # canonical primary label, lower-case
    recommendation_labels: list[str]        # all labels covered by this unit
    heading_text: str                       # e.g. "Recommendation" (keyword found)
    quoted_recommendation_text: str | None  # repeated recommendation text, if found
    response_text: str                      # clean government response
    full_unit_text: str                     # full block (debug)
    page_start: int | None
    page_end: int | None
    char_start: int
    char_end: int
    extraction_confidence: float
    boundary_reason: str  # next_heading | doc_end | multi_label_block


# ---------------------------------------------------------------------------
# Inline split pattern — no line-start requirement
# ---------------------------------------------------------------------------

# Matches every occurrence of "Recommendation N.M" (or Conclusion / Finding)
# anywhere in the flat normalised text.
_INLINE_SPLIT = re.compile(
    r"""
    \b
    (?:government\s+response\s+to\s+)?  # optional prefix (stripped by normaliser in most cases)
    (?:recommendation|conclusion|finding)
    \s+
    (?:
        # Optional section-title words, e.g. "Weight management interventions 8.26".
        # The negative lookahead prevents matching when the first word is a
        # response-status word ("accepted", "rejected", …).  Without this guard
        # the pattern greedily swallows "Recommendation accepted From 9 October"
        # or "Recommendation accepted The minister announced on 8 July" and
        # produces fake labels from the date/ordinal digits.
        (?!(?:accepted|rejected|partially|acknowledged|noted|deferred)\b)
        [A-Za-z]+(?:\s+[A-Za-z]+){0,5}\s+
    )?
    (?P<label>\d{1,3}(?:\.\d{1,3})?[a-z]?)  # "8.1", "1", "4a"
    \b
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Response-opener phrases — signal the start of the actual government reply.
_RESPONSE_OPENER = re.compile(
    r"""
    \b(?:
        the\s+government\s+
            (?:agrees?|accepts?|rejects?|notes?|welcomes?|
               acknowledges?|recognises?|recognizes?|will|has|is|believes?)\b
        |(?:we|the\s+government)\s+
            (?:agree|accept|reject|note|welcome|acknowledge|
               recognise|recognize|will|have|are|believe)\b
        |(?:accepted|partially\s+accepted|rejected|noted)\b
        |the\s+government\s+response\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Strong-form opener: unambiguously institutional/government voice.
# Covers third-person "The Government..." (including negation forms like
# "does not believe") and UK department/agency names as sentence subjects.
# Preferred over "We..." openers in _split_quoted_and_response Pass 1.
_STRONG_RESPONSE_OPENER = re.compile(
    r"""
    \b(?:
        the\s+government\s+
            (?:does\s+not\s+|will\s+not\s+|has\s+not\s+|is\s+not\s+|cannot\s+)?
            (?:agrees?|accepts?|rejects?|notes?|welcomes?|
               acknowledges?|recognises?|recognizes?|will|has|is|believes?|
               does|plans?|intends?|supports?|considers?|expects?|remains?)\b
        |the\s+department\s+for\s+\w+        # "The Department for Transport/Education/..."
        |the\s+home\s+office\b
        |the\s+cabinet\s+office\b
        |hm\s+(?:government|treasury)\b
        |the\s+treasury\b
        # DBT (Department for Business and Trade) — Post Office Horizon IT Inquiry
        |dbt\s+(?:accepts?|agrees?|rejects?|broadly\s+accepts?\b|confirms?|will\b|has\b
                  |notes?|sees?\b|is\b|recognises?|recognizes?|intends?|plans?)\b
        # "The Department [broadly] accepts/rejects/…" without "for <name>"
        |the\s+department\s+(?:broadly\s+)?
            (?:accepts?|rejects?|agrees?|notes?|will\b|has\b|broadly\s+accepts?\b)\b
        |(?:accepted|partially\s+accepted|rejected|noted|acknowledged)\b
        |the\s+government\s+response\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Sentence-split opener: used exclusively in _split_quoted_and_response to locate
# the boundary between the quoted recommendation echo and the government reply.
#
# Stricter than _RESPONSE_OPENER:
#   • Adds institutional subjects (The Department for Transport, Home Office …)
#   • Adds negation forms (The Government does not …, will not …)
#   • Removes committee-voice "we" verbs: note / welcome / are / acknowledge /
#     recognise / recognize — these appear inside recommendation text and must
#     NOT trigger a split inside the quoted echo.
_SENTENCE_SPLIT_OPENER = re.compile(
    r"""
    \b(?:
        # Institutional third-person (unambiguous government voice)
        the\s+government\s+
            (?:does\s+not\s+|will\s+not\s+|has\s+not\s+|is\s+not\s+|cannot\s+)?
            (?:agrees?|accepts?|rejects?|notes?|welcomes?|
               acknowledges?|recognises?|recognizes?|will|has|is|believes?|
               does|plans?|intends?|supports?|considers?|expects?|remains?)\b
        |the\s+department\s+for\s+\w+
        |the\s+home\s+office\b
        |the\s+cabinet\s+office\b
        |hm\s+(?:government|treasury)\b
        |the\s+treasury\b
        # DBT (Department for Business and Trade) — Post Office Horizon IT Inquiry
        |dbt\s+(?:accepts?|agrees?|rejects?|broadly\s+accepts?\b|confirms?|will\b|has\b
                  |notes?|sees?\b|is\b|recognises?|recognizes?|intends?|plans?)\b
        # "The Department [broadly] accepts/rejects/…" without "for <name>"
        |the\s+department\s+(?:broadly\s+)?
            (?:accepts?|rejects?|agrees?|notes?|will\b|has\b|broadly\s+accepts?\b)\b
        # First-person government voice — only unambiguous commitment verbs
        # ("we agree/accept/reject/will/have/believe"). Deliberately excludes
        # note/welcome/are/acknowledge which often appear in committee rec text.
        |(?:we)\s+(?:agree|accept|reject|will|have|believe)\b
        # Formal status markers
        |(?:accepted|partially\s+accepted|rejected|noted|acknowledged)\b
        |the\s+government\s+response\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Minimum chars a unit block must contain to be kept as a structural unit.
# Blocks shorter than this are likely inline references ("see Recommendation 8.1").
_MIN_BLOCK_CHARS = 120

# Maximum chars before the first response opener in which we consider the
# opener paragraph-initial.  Kept generous to handle long quoted recs.
_OPENER_SEARCH_WINDOW = 6000

# If two consecutive label matches are within this many chars of each other
# with no response opener in between, they are considered part of the same
# multi-label block and merged.  Keep this small — genuine multi-label blocks
# have labels right next to each other; anything larger is a separate unit.
_MULTILABEL_MERGE_WINDOW = 50

# Sub-labels: labels that appear inside the quoted portion of a unit block
# without the "Recommendation" keyword prefix (e.g., "8.15. We welcome...").
# Only look for labels whose section matches the primary label's section.
_SUB_LABEL = re.compile(
    r"\b(\d{1,3}\.\d{1,3}[a-z]?)\.\s+[A-Z]",
)

# Known response-intro sentences that precede a "The Government…" opener but
# belong to the response, not the recommendation echo.  Used exclusively inside
# _split_quoted_and_response's backward-expansion step so we don't roll back
# into recommendation echo continuation sentences (e.g. "This should include…").
_RESPONSE_INTRO = re.compile(
    r"""
    \b(?:
        it\s+is\s+essential\s+that\b            # "It is essential that policy making…"
        |as\s+(?:stated|noted)\s+above\b        # "As stated above, it is not a question…"
        |as\s+the\s+committee\s+has\s+acknowledged\b  # govt quoting committee
        |in\s+launching\s+the\b                 # "In launching the Local Sustainable…"
        |to\s+help\s+local\s+areas\b            # "To help local areas successfully tackle…"
        |food\s+labelling\s+is\b                # "Food labelling is an area of EU…"
        |local\s+policy\s+packages\b            # "Local policy packages, such as those…"
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Bare labels at page-join boundaries: "\n 8.32 We recommend..."
# These appear when fitz emits a page number or item label at a \n boundary
# without the "Recommendation" keyword — typically mislabelled final blocks.
_BARE_LABEL_SPLIT = re.compile(
    r"\n\s*(\d{1,3}\.\d{1,3})\s+(?=[A-Z])",
)

# Trailing unnumbered section-heading leaked into the end of a response block.
# Pattern: after sentence-ending punctuation, "Recommendation" with capital R
# followed by a capital title word and no digit — i.e. a thematic/section heading
# such as "Recommendation Promoting and enabling choice: the role of regulation".
# The capital-R + capital-word requirement ensures:
#   • "the Committee's recommendation" (lower-r) — NOT matched → safe
#   • "House of Lords recommendations" (lower-r, plural) — NOT matched → safe
#   • "Recommendation 8.32 ..." (digit after keyword) — capital R but [A-Z] fails → safe
_TRAILING_REC_HEADING = re.compile(
    r"(?<=[.!?])\s+Recommendation\s+[A-Z][^\n]*$"
)

# Post Office Horizon IT Inquiry — explicit status / classification line that
# appears between the quoted recommendation text and the government response body.
# Examples: "Recommendation accepted", "Recommendation rejected",
#           "Recommendation broadly accepted", "Recommendation acknowledged".
# Using this as a high-confidence split point is safer than any opener pattern
# because the status line is always right at the rec/response boundary.
_PO_STATUS = re.compile(
    r"\bRecommendation\s+"
    r"(?:accepted|rejected|partially\s+accepted|broadly\s+accepts?\b|acknowledged|noted|deferred)\b",
    re.IGNORECASE,
)

# Committee (inquiry) voice patterns — unambiguously from the SELECT COMMITTEE,
# never from the government response.  Used to detect the end of the
# recommendation echo so that introductory response paragraphs that precede
# the first "The Government…" sentence are not discarded.
#
# Deliberately EXCLUDES "we note", "we acknowledge", "we welcome" alone because
# these also appear as government response phrases ("We note the Committee's
# recommendation…").  The committee-specific multi-word forms
# ("although we welcome", "whilst we welcome") are included.
_COMMITTEE_REC_VOICE = re.compile(
    r"""
    \b(?:
        # Directive verbs — only the committee recommends/urges/invites
        we\s+(?:therefore\s+|strongly\s+|further\s+|now\s+|also\s+)?
            (?:recommend\b|urge\b|invite\b|draw\s+attention\b|ask\s+that\b|
               call\s+on\b|encourage\b)
        # Concessive constructions clearly scoped to the committee
        |although\s+we\s+(?:welcome|note|recognise|recognize)\b
        |whilst\s+we\s+(?:welcome|note|recognise|recognize)\b
        |while\s+we\s+(?:welcome|note|recognise|recognize)\b
        # Editorial inference phrases used in committee reports
        |this\s+suggests?\s+that\b
        |this\s+would\s+(?:jeopardise|jeopardize)\b
        # Back-reference to the committee's own text
        |as\s+we\s+have\s+(?:noted|observed|recommended)\b
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_label(raw: str) -> str:
    return raw.strip().lower().rstrip(".")


def _page_for_offset(
    char_offset: int,
    page_breaks: list[tuple[int, int | None]],
) -> int | None:
    result: int | None = None
    for start, pno in page_breaks:
        if char_offset >= start:
            result = pno
        else:
            break
    return result


def _is_para_initial(start: int, block: str) -> bool:
    """Return True if position *start* is paragraph-initial in *block*.

    Uses a 30-char lookback so sentences preceded by many spaces (collapsed
    from newlines by the normaliser) are still detected as paragraph-initial.
    An 8-char window misses the period when e.g. "interventions.           The"
    has 11 spaces between the period and the opener keyword.
    """
    if start == 0:
        return True
    prefix = block[max(0, start - 40): start]
    if re.search(r"\n[ \t]*\n[ \t]*\Z", prefix):
        return True
    if re.search(r"[.!?]\s+\Z", prefix):
        return True
    if re.search(r"\n[ \t]*\Z", prefix):
        return True
    # Allow an inline parenthetical reference between the previous sentence and
    # the opener, e.g. "...be reconsidered. (113.83) The government accepts...".
    # Grenfell Phase 2 uses these (paragraph.line) citations between sentences.
    if re.search(r"[.!?]\s+\([\d\.\s\-,]+\)\s+\Z", prefix):
        return True
    return False


def _find_echo_end(block: str, after_pos: int = 0) -> int:
    """Return the char position immediately after the last committee-voice
    recommendation sentence that starts at or after *after_pos* in *block*.

    Returns *after_pos* unchanged when no committee-voice sentence is found,
    so callers can detect the "nothing found" case with ``result == after_pos``.
    """
    last_end = after_pos
    for m in _COMMITTEE_REC_VOICE.finditer(block, after_pos):
        # Find the end of the sentence that contains this match.
        sent_end = re.search(r"[.!?]", block[m.end(): m.end() + 800])
        pos = m.end() + (sent_end.end() if sent_end else 0)
        if pos > last_end:
            last_end = pos
    return last_end


def _split_quoted_and_response(block: str) -> tuple[str | None, str]:
    """
    Split a unit block into (quoted_recommendation_text, response_text).

    For grouped recommendation blocks (e.g., 8.14 / 8.15 / 8.16 sharing one
    government reply), sub-labels like "8.15. We welcome..." appear inside the
    quoted echo.  The opener "We welcome" would otherwise look paragraph-initial
    because it is preceded by "8.15. " which ends with ". ".

    Strategy
    --------
    1. Find the last sub-label position (e.g., "8.16.") in the block using
       _SUB_LABEL.  This marks the end of the recommendation echo region.
    2. Prefer STRONG openers ("The Government agrees/accepts/...") that appear
       after the last sub-label.
    3. Fall back to any _RESPONSE_OPENER after the last sub-label that is
       paragraph-initial AND not immediately following a numbered label (300-char
       prefix guard).
    4. Final fallback: original paragraph-initial logic with no positional guard
       (for blocks that have no sub-labels at all).
    """
    if not block:
        return None, ""
    window = block[:_OPENER_SEARCH_WINDOW]

    # Use the stricter _SENTENCE_SPLIT_OPENER for candidate detection.
    # _RESPONSE_OPENER is deliberately kept broader for structural detection
    # (Step 2) and multi-label merging (Step 3), but must not drive splitting
    # because it includes "we note/welcome/are/acknowledge" which appear as
    # committee voice inside the recommendation echo.
    candidates = list(_SENTENCE_SPLIT_OPENER.finditer(window))
    # Do NOT early-return when candidates is empty — Pass 5 (committee-echo-end
    # fallback) can still find a valid split point without any
    # _SENTENCE_SPLIT_OPENER match.

    # Determine the start position of the last sub-label in the block.
    # If sub-labels exist, openers before or at that position are inside the
    # grouped recommendation echo and must be skipped.
    last_sub_label_start: int = -1
    for m in _SUB_LABEL.finditer(window):
        last_sub_label_start = m.start()

    chosen: int | None = None

    # ── Pass 0: document-specific status-line split ─────────────────────────
    # Response documents that use an explicit "Recommendation accepted /
    # rejected / partially accepted / acknowledged" status line (e.g. Post
    # Office Horizon IT Inquiry) place it right between the recommendation text
    # and the government response body.  Splitting immediately AFTER the status
    # phrase gives the most reliable start boundary for this document type and
    # is safer than any opener-word heuristic.
    # _is_para_initial guards against matching "recommendation accepted" that
    # appears inline inside a sentence (e.g. "…whose recommendation accepted
    # the proposal…").
    for m in _PO_STATUS.finditer(window):
        if _is_para_initial(m.start(), block):
            ws = re.match(r"\s*", block[m.end():])
            candidate = m.end() + (ws.end() if ws else 0)
            if candidate < len(block):
                chosen = candidate
                break

    if not candidates:
        # Skip passes 1–4; Pass 5 (echo-end fallback) below will handle it.
        pass

    # ── Pass 1: strong opener after last sub-label, paragraph-initial ──────
    if chosen is None:
        for m in candidates:
            start = m.start()
            if last_sub_label_start >= 0 and start <= last_sub_label_start:
                continue
            if not _STRONG_RESPONSE_OPENER.match(m.group()):
                continue
            if _is_para_initial(start, block):
                chosen = start
                break

    # ── Pass 2: any opener after last sub-label, with numbered-label guard ─
    # Skip openers that are still inside the recommendation echo (detected by
    # a numbered label in the preceding 300 chars).
    if chosen is None:
        for m in candidates:
            start = m.start()
            if last_sub_label_start >= 0 and start <= last_sub_label_start:
                continue
            if not _is_para_initial(start, block):
                continue
            wide_prefix = block[max(0, start - 300): start]
            if re.search(r"\b\d{1,3}\.\d{1,3}\.", wide_prefix):
                continue  # still inside recommendation echo
            chosen = start
            break

    # ── Pass 3: relax numbered-label guard but keep sub-label position guard ─
    if chosen is None:
        for m in candidates:
            start = m.start()
            if last_sub_label_start >= 0 and start <= last_sub_label_start:
                continue
            if _is_para_initial(start, block):
                chosen = start
                break

    # ── Pass 4: original logic, no positional constraints (last resort) ────
    if chosen is None:
        for m in candidates:
            start = m.start()
            if _is_para_initial(start, block):
                chosen = start
                break

    # ── Backward expansion ──────────────────────────────────────────────────
    # If passes 1-4 found a split at X but the response actually starts earlier
    # (because the block opens with a non-government-voice intro paragraph),
    # look for a _RESPONSE_INTRO sentence between the end of the last
    # committee-voice sentence and X.  If found, roll the split back to that
    # intro sentence.
    #
    # Using _RESPONSE_INTRO (rather than a generic "no committee-voice" test)
    # prevents rolling back into recommendation echo continuation sentences
    # such as "This archive should provide accounts…" or "This should include
    # consideration of…".
    if chosen is not None:
        search_start = last_sub_label_start + 1 if last_sub_label_start >= 0 else 0
        echo_end = _find_echo_end(block, search_start)
        if echo_end > search_start and echo_end < chosen - 30:
            between = block[echo_end:chosen]
            intro_m = _RESPONSE_INTRO.search(between)
            if intro_m:
                candidate = echo_end + intro_m.start()
                if _is_para_initial(candidate, block):
                    chosen = candidate

    # ── Pass 5: committee-echo-end fallback ─────────────────────────────────
    # For blocks where the response never starts with a recognised opener
    # (e.g. "Food labelling is an area of EU competence…", "As the Committee
    # has acknowledged…", "Local policy packages…", "To help local areas…"),
    # fall back to splitting immediately after the last committee-voice sentence.
    if chosen is None:
        search_start = last_sub_label_start + 1 if last_sub_label_start >= 0 else 0
        echo_end = _find_echo_end(block, search_start)
        if echo_end > search_start:
            ws = re.match(r"\s*", block[echo_end:])
            candidate = echo_end + (ws.end() if ws else 0)
            if candidate < len(block):
                chosen = candidate

    if chosen is None:
        return None, block.strip()

    quoted = block[:chosen].strip() or None
    response = block[chosen:].strip()
    return quoted, response


def _find_sub_labels(quoted_text: str | None, primary_label: str) -> list[str]:
    """
    Scan *quoted_text* for inline sub-labels belonging to the same section
    as *primary_label* (e.g., primary "8.14" → finds "8.15", "8.16").

    Sub-labels look like "8.15. We welcome..." — a dotted number followed
    by ". " and an uppercase letter, with no "Recommendation" keyword.
    Returns a list of additional normalised labels (not including primary).
    """
    if not quoted_text or "." not in primary_label:
        return []
    try:
        section = primary_label.split(".")[0]
    except Exception:
        return []
    found: list[str] = []
    for m in _SUB_LABEL.finditer(quoted_text):
        raw = m.group(1)
        if raw.startswith(section + ".") and raw != primary_label:
            norm = raw.strip().lower().rstrip(".")
            if norm not in found:
                found.append(norm)
    return found


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _extract_units_by_simple_blocks(
    full_text: str,
    page_breaks: list[tuple[int, int | None]],
    source: str,
) -> list[ResponseUnit]:
    """
    Fallback parser for response documents that use a simple
    "Recommendation: <text> Response: <text>" block structure without
    numbered "Recommendation N" headings (e.g. Summer 2024 Disorder
    response).

    Each Recommendation: / Response: pair becomes one unit, labelled by
    its sequential position (1, 2, 3, …) so the matcher can pair it with
    the corresponding ordinal recommendation in the policy document.
    """
    rec_re = re.compile(r"\bRecommendation:\s*", re.IGNORECASE)
    resp_re = re.compile(r"\bResponse:\s*", re.IGNORECASE)

    rec_positions = [m for m in rec_re.finditer(full_text)]
    if len(rec_positions) < 2:
        return []

    units: list[ResponseUnit] = []
    for idx, rec_m in enumerate(rec_positions):
        block_start = rec_m.end()
        block_end = rec_positions[idx + 1].start() if idx + 1 < len(rec_positions) else len(full_text)
        block = full_text[block_start:block_end]

        # Split into (quoted recommendation, response) on the first "Response:"
        resp_m = resp_re.search(block)
        if resp_m:
            quoted = block[:resp_m.start()].strip()
            response_text = block[resp_m.end():].strip()
        else:
            # No "Response:" inside the block — likely an in-text reference.
            # Skip unless block is long enough to be a standalone response.
            if len(block.strip()) < _MIN_BLOCK_CHARS:
                continue
            quoted = ""
            response_text = block.strip()

        # Trim a trailing topic-heading carry-over ("5. The political response")
        # that leaks into the response when the next block starts mid-sentence.
        response_text = re.sub(
            r"\s+\d+\.\s+[A-Z][^.]{0,80}$", "", response_text
        ).strip()

        if not response_text:
            continue

        label = str(idx + 1)
        units.append({
            "unit_id": idx,  # 0-indexed — used as np array index in alignment.py
            "source": source,
            "recommendation_label": label,
            "recommendation_labels": [label],
            "heading_text": "Recommendation",
            "quoted_recommendation_text": quoted or None,
            "response_text": response_text,
            "full_unit_text": full_text[rec_m.start():block_end].strip(),
            "page_start": _page_for_offset(rec_m.start(), page_breaks),
            "page_end": _page_for_offset(block_end - 1, page_breaks),
            "char_start": rec_m.start(),
            "char_end": block_end,
            "extraction_confidence": 0.85,
            "boundary_reason": "simple_block_pair",
        })
    return units


_BLOOD_STATUS_SPLIT = re.compile(
    r"""
    \b(?:
        (?:this|these)\s+recommendations?
        |recommendations?\s+\d{1,2}\)?\s*(?:[a-z](?:-[a-z])?\)?)?
    )
    \s+(?:is|are)\s+accepted\b[^.]*\.
    """,
    re.IGNORECASE | re.VERBOSE,
)

_BLOOD_RESPONSE_GROUPS: list[tuple[list[str], str]] = [
    (["1"], r"\b1\)\s+Compensation Scheme\b"),
    (["2a", "2b", "2c"], r"\b2\)\s+Recognising and remembering\b"),
    (["3a", "3b"], r"\b3\)\s+Learning from the Inquiry\b"),
    (["3c"], r"\bc\.\s+The Inquiry website is maintained online\."),
    (["4a i", "4a ii", "4a iii"], r"\b4a\)\s+i-iii\)\s+Duty of Candour\b"),
    (["4a iv", "4a v"], r"\b4a\)\s+iv-v\)\s+statutory duty of candour\b"),
    (["4b", "4c i", "4c ii"], r"\b4b\)\s+Cultural Change and 4c\)\s+i-ii\)\s+Regulation\b"),
    (["4d"], r"\b4d\)\s+Patient Records\b"),
    (["4e"], r"\b4e\)\s+Coordination of patient records\b"),
    (["5a", "5b", "5c"], r"\b5\)\s+Ending the Defensive Culture\b"),
    (["6a i", "6a ii", "6a iii", "6a iv", "6a v", "6a vi"], r"\b6\)\s+Monitoring Liver damage\b"),
    (["7a i", "7a ii", "7a iii"], r"\b7\)\s+Patient Safety:\s+Blood Transfusions\b"),
    (["7b", "7c", "7d", "7e"], r"\b7b\)\s+Review of progress\b"),
    (["7f i", "7f ii", "7f iii"], r"\b7f\)\s+Establishing the outcome\b"),
    (["8a", "8b"], r"\b8\)\s+Finding the undiagnosed\b"),
    (["9a", "9b", "9c", "9d", "9e", "9f"], r"\b9\)\s+Protecting the Safety\b"),
    (["10a i", "10a ii", "10a iii"], r"\b10\)\s+Giving patients a voice\b"),
    (["10a iv"], r"\b10a\)\s+iv\)\s+Thalassaemia and Sickle Cell\b"),
    (["10a v"], r"\b10a\)\s+v\)\s+Patient Feedback\b"),
    (["11a", "11b", "11c", "11d"], r"\b11\)\s+Responding to calls\b"),
    (["12a", "12b", "12c"], r"\b12\)\s+Giving effect to the recommendations\b"),
    (["12d", "12e"], r"\bRecommendations\s+12\)\s+d-e\)\s+are\s+accepted\b"),
]


def _split_blood_quoted_and_response(block: str) -> tuple[str | None, str]:
    """Split an Infected Blood grouped section at its explicit status line."""
    status = _BLOOD_STATUS_SPLIT.search(block)
    if not status:
        return None, block.strip()
    quoted = block[: status.start()].strip() or None
    response = block[status.start():].strip()
    return quoted, response


def _extract_infected_blood_response_units(
    full_text: str,
    page_breaks: list[tuple[int, int | None]],
    source: str,
) -> list[ResponseUnit]:
    """
    Parse the Infected Blood response's grouped recommendation sections.

    The response is structured by recommendation headings rather than repeated
    "Recommendation N:" headings in the body. Child recommendations are kept as
    labels on grouped units so downstream alignment preserves 2a, 4a i, 12e,
    etc. without falling back to arbitrary chunks.
    """
    positions: list[tuple[int, int, list[str], str]] = []
    for labels, pattern in _BLOOD_RESPONSE_GROUPS:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if not match:
            continue
        positions.append((match.start(), match.end(), labels, match.group(0)))

    positions.sort(key=lambda item: item[0])
    if len(positions) < 8:
        return []

    units: list[ResponseUnit] = []
    for idx, (start, _heading_end, labels, heading) in enumerate(positions):
        end = positions[idx + 1][0] if idx + 1 < len(positions) else len(full_text)
        if labels == ["12d", "12e"]:
            useful_links = re.search(r"\bUseful Links\b", full_text[start:end], re.IGNORECASE)
            if useful_links:
                end = start + useful_links.start()
        block = full_text[start:end].strip()
        if not block:
            continue

        quoted, response_text = _split_blood_quoted_and_response(block)
        response_text = re.sub(r"\s*\bUseful Links\b[\s\S]*$", "", response_text, flags=re.IGNORECASE).strip()
        if not response_text:
            response_text = block

        boundary_reason = "structured_grouped" if len(labels) > 1 else "structured_section"
        units.append(
            ResponseUnit(
                unit_id=len(units),
                source=source,
                recommendation_label=labels[0],
                recommendation_labels=labels,
                heading_text=heading.strip(),
                quoted_recommendation_text=quoted,
                response_text=response_text,
                full_unit_text=block,
                page_start=_page_for_offset(start, page_breaks),
                page_end=_page_for_offset(max(end - 1, start), page_breaks),
                char_start=start,
                char_end=end,
                extraction_confidence=0.9,
                boundary_reason=boundary_reason,
            )
        )
    return units


def extract_response_units(response_pages: list[PageRecord]) -> list[ResponseUnit]:
    """
    Split a government response document into per-recommendation response units.

    Works on flat normalised text (single newlines collapsed to spaces) as
    well as on text that preserves paragraph breaks.

    Returns an empty list when no structural recommendation headings are
    detected so the caller can fall back to chunk alignment.
    """
    if not response_pages:
        return []

    source = response_pages[0]["source"]

    # Build joined text, tracking page start offsets.
    parts: list[str] = []
    page_breaks: list[tuple[int, int | None]] = []
    cursor = 0
    for page in response_pages:
        page_breaks.append((cursor, page["page_number"]))
        page_text = page["text"]
        parts.append(page_text)
        cursor += len(page_text) + 1  # +1 for the '\n' join separator

    full_text = "\n".join(parts)

    if "Volume_1-Blood-Inquiry-Response" in source:
        blood_units = _extract_infected_blood_response_units(full_text, page_breaks, source)
        if blood_units:
            return blood_units

    # ── Step 1: find all inline "Recommendation N.M" occurrences ─────────
    all_matches = list(_INLINE_SPLIT.finditer(full_text))
    if not all_matches:
        # Fallback: documents without numeric "Recommendation N" headings may
        # still use a "Recommendation: … Response: …" block structure.
        return _extract_units_by_simple_blocks(full_text, page_breaks, source)

    # ── Step 1b: supplementary bare-label pass ────────────────────────────
    # Some response PDFs mislabel a final block with just "\n 8.32 We recommend..."
    # — no "Recommendation" keyword.  These appear at page-join \n boundaries.
    # Collect them so they participate in candidate filtering below.
    bare_label_extras: list[dict] = []
    existing_starts = {m.start() for m in all_matches}
    for bm in _BARE_LABEL_SPLIT.finditer(full_text):
        # The group(0) starts at \n; the label text begins at group(1).
        label_char_start = bm.start(1)
        # Only add if not already covered by _INLINE_SPLIT within ±30 chars.
        if any(abs(label_char_start - es) < 30 for es in existing_starts):
            continue
        bare_label_extras.append({
            "match_start": bm.start(),   # \n position (structural boundary)
            "label_start": bm.end(),      # text starts after the label token
            "raw_label": bm.group(1),
            "heading": "Recommendation",
            "bare_label": True,
        })

    # ── Step 2: determine structural split points ─────────────────────────
    # A match is "structural" if the block it introduces (text from this
    # match to the next) contains a response opener OR is long enough to be
    # a real recommendation block (not a brief inline citation).
    #
    # We work in two passes:
    #   A) assign each match a candidate block end = next match's start.
    #   B) filter out matches whose block has no opener AND is short.

    # Build candidate (start, end, label) triples.
    candidates: list[dict] = []
    for i, m in enumerate(all_matches):
        # Exclude cross-reference citations like "under recommendation 8.30"
        # or "see recommendation 8.1" — these are inline, not structural headings.
        match_pre = full_text[max(0, m.start() - 20): m.start()].lower()
        if re.search(r"\b(?:under|see|per)\s*\Z", match_pre):
            continue

        # ── Integer-label cross-ref / TOC filter ─────────────────────────
        # Documents like the Post Office Horizon IT Inquiry use integer labels
        # (1–19) rather than decimal ones (8.1).  Their front matter contains a
        # table-of-contents listing ("Recommendation 1 Recommendation 2 …") and
        # their response paragraphs contain many cross-references such as
        # "…accepting Recommendation 9, this interpretation…".
        #
        # For integer-only labels, a structural heading ALWAYS begins a fresh
        # sentence, so the text immediately after the label must:
        #   (a) start with a capital letter — otherwise it is a sentence
        #       continuation (cross-reference ending with "," or "." or
        #       starting with a lowercase word), AND
        #   (b) NOT start with another "Recommendation" keyword — otherwise
        #       it is a table-of-contents listing with no content between entries.
        #
        # This filter is intentionally skipped for decimal labels (e.g. "8.1")
        # because those documents (BC, Space Economy, Grenfell …) have their own
        # label notation where a trailing "." is part of the label ("8.1.") and
        # structural headings legitimately look like "8.1. The idea…".
        raw_lbl = m.group("label")
        if "." not in raw_lbl:
            post_text_raw = full_text[m.end(): m.end() + 60].lstrip()
            # Some docs (Covid 1/2, Blood, Grenfell) use a colon between the
            # label and the section title: "Recommendation 1: A simplified…".
            # Strip ONLY a leading colon — a leading period signals end-of-
            # sentence citation ("…as accepted in Recommendation 2. DBT's
            # view…") and must NOT be treated as a heading separator.
            post_text = re.sub(r"^:\s*", "", post_text_raw)
            # (b) TOC entry — immediately followed by another recommendation keyword
            if re.match(r"Recommendation\b", post_text, re.IGNORECASE):
                continue
            # (a) Cross-reference — text continues mid-sentence (punctuation or
            # lowercase) rather than starting a new sentence.
            starts_with_subitem = bool(
                re.match(r"(?:\(?[a-z]\)|[a-z][.)])\s+", post_text, re.IGNORECASE)
            )
            if not re.match(r"[A-Z]", post_text) and not starts_with_subitem:
                continue
            # (c) TOC entry — leader-dot run inside the next 300 chars marks
            # a contents-list line ("Recommendation 1: title .............7
            # Recommendation 2: …").  Genuine body sections never contain
            # \.{5,} runs, so this is a clean TOC signal.
            peek = full_text[m.end(): m.end() + 300]
            if re.search(r"\.{5,}", peek):
                continue
            # (d) TOC entry — another "Recommendation N:" within 200 chars
            # with no response opener between.  Catches docs whose TOC has
            # page ranges (e.g. "Recommendation 1: Compensation Scheme 15-24
            # Recommendation 2: Recognising …") rather than leader dots.
            # 200-char window keeps very-short legitimate body sections
            # (e.g. Covid 2 rec 13: a one-sentence "not for the UK government
            # to respond" reply that is ~370 chars before rec 14) intact.
            peek_short = full_text[m.end(): m.end() + 200]
            next_rec = re.search(r"\bRecommendation\s+\d+[a-z]?:", peek_short)
            if next_rec and not _RESPONSE_OPENER.search(peek_short[:next_rec.start()]):
                continue
            # Passed all structural gates → genuine heading.  Skip the
            # block-content check below: the next raw match (all_matches[i+1])
            # may be a nearby cross-reference whose tiny gap would wrongly fail
            # the is_long test (e.g. "Recommendation 11 The 'best offer'…" is
            # only ~83 chars from the cross-ref "Recommendation 10, shall be…").
            candidates.append({
                "match_start": m.start(),
                "label_start": m.end(),
                "raw_label": raw_lbl,
                "heading": "Recommendation",
            })
            continue

        block_start = m.end()
        block_end = all_matches[i + 1].start() if i + 1 < len(all_matches) else len(full_text)
        block_text = full_text[block_start:block_end]
        has_opener = bool(_RESPONSE_OPENER.search(block_text))
        is_long = (block_end - block_start) >= _MIN_BLOCK_CHARS
        if not has_opener and not is_long:
            # Likely an inline citation — skip.
            continue
        candidates.append({
            "match_start": m.start(),
            "label_start": block_start,
            "raw_label": m.group("label"),
            "heading": "Recommendation",
        })

    # Inject bare-label extras that fall after the last structural candidate.
    # The cross-reference filter above ensures last_candidate_end refers to a
    # genuine heading, not a false citation like "under recommendation 8.30".
    if bare_label_extras:
        last_candidate_end = candidates[-1]["match_start"] if candidates else 0
        for extra in bare_label_extras:
            if extra["match_start"] <= last_candidate_end:
                continue  # already covered by a genuine _INLINE_SPLIT candidate
            block_text = full_text[extra["label_start"]:]
            has_opener = bool(_RESPONSE_OPENER.search(block_text[:_OPENER_SEARCH_WINDOW]))
            is_long = len(block_text) >= _MIN_BLOCK_CHARS
            if has_opener or is_long:
                extra_entry = dict(extra)
                extra_entry["boundary_reason_override"] = "bare_label_at_boundary"
                candidates.append(extra_entry)
        candidates.sort(key=lambda c: c["match_start"])

    if not candidates:
        return []

    # ── Step 3: merge multi-label blocks ─────────────────────────────────
    # If two consecutive structural candidates are within _MULTILABEL_MERGE_WINDOW
    # chars of each other AND the text between them has no response opener,
    # they share one government reply → merge labels under the first.

    merged: list[dict] = []
    i = 0
    while i < len(candidates):
        current = dict(candidates[i])
        current["labels"] = [_normalize_label(current["raw_label"])]
        j = i + 1
        while j < len(candidates):
            gap = candidates[j]["match_start"] - candidates[i]["match_start"]
            if gap > _MULTILABEL_MERGE_WINDOW:
                break
            gap_text = full_text[candidates[i]["label_start"] : candidates[j]["match_start"]]
            if _RESPONSE_OPENER.search(gap_text):
                break  # response already started — next candidate is its own unit
            next_lbl = _normalize_label(candidates[j]["raw_label"])
            if next_lbl not in current["labels"]:
                current["labels"].append(next_lbl)
            j += 1
        current["next_idx"] = j  # the next unmerged candidate index
        merged.append(current)
        i = j

    # ── Step 4: build ResponseUnit objects ────────────────────────────────
    units: list[ResponseUnit] = []
    for idx, entry in enumerate(merged):
        block_start = entry["label_start"]
        if idx + 1 < len(merged):
            block_end = merged[idx + 1]["match_start"]
            boundary_reason = "next_heading"
        else:
            block_end = len(full_text)
            boundary_reason = "doc_end"

        is_multi = len(entry["labels"]) > 1
        if is_multi:
            boundary_reason = "multi_label_block"

        # Propagate override for bare-label units.
        if entry.get("boundary_reason_override"):
            boundary_reason = entry["boundary_reason_override"]

        full_block = full_text[block_start:block_end].strip()
        quoted, response_text = _split_quoted_and_response(full_block)
        response_text = _TRAILING_REC_HEADING.sub("", response_text).strip()
        # Strip trailing chapter-heading leak ("Chapter 2: Action taken since
        # 2017 …") which appears when the last response unit in a document
        # absorbs the start of the next chapter due to no following label.
        response_text = re.sub(
            r"\s*\bChapter\s+\d+\b[\s\S]*$", "", response_text
        ).strip()

        # ── Sub-label detection ─────────────────────────────────────────────
        # Scan the quoted portion for inline sub-labels (e.g., "8.15. We welcome...")
        # that share the same section as the primary label but have no
        # "Recommendation" keyword.  These are additional recs covered by this block.
        primary_label = entry["labels"][0]
        sub_labels = _find_sub_labels(quoted, primary_label)
        # Safety net: when the split didn't produce a quoted portion (quoted is
        # None), scan the full block so sub-labels are not missed entirely.
        if not sub_labels and quoted is None:
            sub_labels = _find_sub_labels(full_block, primary_label)
        all_labels = list(entry["labels"])
        for sl in sub_labels:
            if sl not in all_labels:
                all_labels.append(sl)
        if len(all_labels) > 1:
            is_multi = True
            if boundary_reason not in ("bare_label_at_boundary",):
                boundary_reason = "multi_label_block"

        page_start = _page_for_offset(entry["match_start"], page_breaks)
        page_end = _page_for_offset(max(block_end - 1, entry["match_start"]), page_breaks)

        units.append(
            ResponseUnit(
                unit_id=idx,
                source=source,
                recommendation_label=primary_label,
                recommendation_labels=all_labels,
                heading_text=entry["heading"],
                quoted_recommendation_text=quoted,
                response_text=response_text,
                full_unit_text=full_block,
                page_start=page_start,
                page_end=page_end,
                char_start=entry["match_start"],
                char_end=block_end,
                extraction_confidence=0.75 if is_multi else 0.95,
                boundary_reason=boundary_reason,
            )
        )

    # ── Sequence correction ───────────────────────────────────────────────────
    # Some response PDFs mislabel a bare-label block with the same number as the
    # preceding unit (e.g., two blocks both labelled "8.32" when the second
    # should be "8.33").  Detect duplicate primary labels and increment them.
    seen_primaries: set[str] = set()
    for unit in units:
        primary = unit["recommendation_label"] or ""
        if primary in seen_primaries and "." in primary:
            try:
                major, minor_str = primary.split(".", 1)
                new_primary = f"{major}.{int(minor_str) + 1}"
                unit["recommendation_label"] = new_primary
                labels = unit.get("recommendation_labels") or []
                unit["recommendation_labels"] = [
                    new_primary if lbl == primary else lbl for lbl in labels
                ] or [new_primary]
                unit["boundary_reason"] = "sequence_correction"
            except (ValueError, AttributeError):
                pass
        seen_primaries.add(unit.get("recommendation_label") or "")

    return units


# ---------------------------------------------------------------------------
# Boundary trimming for the chunk-fallback path
# ---------------------------------------------------------------------------

def trim_to_response_boundary(text: str) -> str:
    """
    Trim *text* so it ends before the next inline "Recommendation N.M"
    heading, to prevent chunk-fallback results from spanning multiple recs.
    """
    if not text:
        return text
    for m in _INLINE_SPLIT.finditer(text):
        start = m.start()
        if start < 60:
            continue  # skip the heading that starts this chunk itself
        # Only trim at what looks like a structural heading (not an inline cite).
        prefix = text[max(0, start - 5) : start]
        if re.search(r"[.!?\n]\s*\Z", prefix):
            return text[:start].rstrip()
    return text
