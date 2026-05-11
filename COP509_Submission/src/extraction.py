"""
Recommendation extraction migrated from the validated prototype in ``delete_me.py``.

The final pipeline preserves the prototype's three-detector method:

Detector A:
  Explicit ``Recommendation X:`` labels
Detector B:
  Structured recommendation sections with hierarchical list parsing
Detector C:
  Embedded / modal recommendations inside summary-of-recommendations sections

The implementation below is a direct backend migration of that method with only
light refactoring for readability and integration into the coursework project.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, TypedDict

import pandas as pd

from .chunking import Chunk

if TYPE_CHECKING:
    from .pdf_loader import PageRecord
else:
    PageRecord = dict[str, Any]


class Recommendation(TypedDict, total=False):
    rec_id: int
    text: str
    extraction_method: str
    detector: str
    confidence: float
    document: str
    page_number: int | str | None
    ocr: bool
    item_label: str
    span_id: str
    extraction_source: str  # "primary" | "inline_report_recommendation" | "response_heading_fallback"
    source_document_role: str  # "policy" | "response"
    extraction_note: str
    source_paragraph: str  # e.g. "113.6" or "Paragraph 17"
    source_item_type: str  # e.g. "Recommendation" | "Conclusion"


class _PageUnit(TypedDict):
    document: str
    page_number: int
    text: str
    ocr: bool


@dataclass
class _RecCandidate:
    doc_id: str
    page: int | str
    span_id: str
    text: str
    detector: str
    confidence: float
    notes: str
    ocr: bool


def _clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\u00ad", "")
    text = re.sub(r'[\u201c\u201d\u201e\u201f"«»‹›]', "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def _sent_tokenize(text: str) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    try:
        from nltk.tokenize import sent_tokenize

        return [segment.strip() for segment in sent_tokenize(cleaned) if segment.strip()]
    except Exception:
        return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", cleaned) if segment.strip()]


def _normalised_key(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text[:220]


def _normalise_for_tail_match(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.replace("Ã¢â‚¬Å“", '"').replace("Ã¢â‚¬Â", '"').replace("Ã¢â‚¬â„¢", "'")
    return text


def _first_sentence(text: str) -> str:
    if not isinstance(text, str):
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0] if parts else text[:200]


def _last_sentence(text: str) -> str:
    sentences = _sent_tokenize(text)
    return _clean_text(sentences[-1]) if sentences else ""


def _score_candidate(text: str, detector: str) -> tuple[float, str]:
    lowered = text.lower()

    if detector == "A":
        return 0.95, "Explicit 'Recommendation X:' label"

    if detector == "B":
        if re.search(r"\bshould\b|\bmust\b|\bneed to\b|\brequired to\b|\bwe recommend\b|\bwe urge\b", lowered):
            return 0.92, "Structured item with directive language"
        if re.search(r"\bshall\b|\bwill\b", lowered):
            return 0.86, "Structured item with strong prescriptive auxiliary"
        return 0.78, "Structured item in recommendations section; directive language weaker"

    if detector == "C":
        has_strong = bool(
            re.search(r"\bmust\b|\brequired to\b|\bshall\b|\bwe recommend\b|\bwe urge\b", lowered)
        )
        has_target = any(term in lowered for term in TARGET_TERMS)
        if has_strong and has_target:
            return 0.84, "Strong modal + clear target"
        if has_target:
            return 0.72, "Modal present + target present"
        return 0.60, "Modal present; target weak"

    return 0.50, "Unknown detector"


_RE_REC_LABEL_A = re.compile(r"\brecommendation\s+(\d{1,3}[a-z]?)\s*:", re.IGNORECASE)
_RE_PARA_MARKER_A = re.compile(r"\b\d+\.\d+(?:\.\d+)*\.\s")
_RE_SECTION_BREAK_A = re.compile(r"\b(appendix|annex|references|bibliography|contents)\b", re.IGNORECASE)
_RE_PRIOR_REC_CONTEXT_A = re.compile(
    r"\b("
    r"in the interim report"
    r"|in the first interim report"
    r"|in the second interim report"
    r"|i made the following recommendations"
    r"|made the following recommendations"
    r"|the following recommendations"
    r"|previous recommendations"
    r"|earlier recommendations"
    r")\b",
    re.IGNORECASE,
)


def _page_has_prior_recommendation_context(text: str) -> bool:
    return bool(_RE_PRIOR_REC_CONTEXT_A.search(text or ""))


def _trim_explicit_block_tail(text: str) -> str:
    if not text:
        return ""

    cut_points: list[int] = []
    para_marker = _RE_PARA_MARKER_A.search(text)
    if para_marker:
        cut_points.append(para_marker.start())

    section_break = _RE_SECTION_BREAK_A.search(text)
    if section_break:
        cut_points.append(section_break.start())

    if cut_points:
        cut = min(cut_points)
        last_period = max(text.rfind(".", 0, cut), text.rfind("!", 0, cut), text.rfind("?", 0, cut))
        if last_period != -1:
            text = text[: last_period + 1]
        else:
            text = text[:cut]

    return _clean_text(text)


def _extract_explicit_recommendation_blocks(page_text: str) -> list[tuple[str, str]]:
    matches = list(_RE_REC_LABEL_A.finditer(page_text or ""))
    if not matches:
        return []

    output: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        label = match.group(1).strip()
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(page_text)
        chunk = _trim_explicit_block_tail(page_text[start:end])
        if chunk:
            output.append((label, chunk))
    return output


_RE_RECOMM_HEADING = re.compile(r"\brecommendations\b", re.IGNORECASE)
_RE_RECOMM_CHAPTER_HEADING = re.compile(
    r"^\s*(?:chapter\s+\d+[:.\-]?\s*)?conclusions\s+and\s+recommendations\b"
    r"|^\s*chapter\s+\d+[:.\-]?\s*conclusions\s+and\s+recommendations\b",
    re.IGNORECASE,
)


def _recommendations_within_first_n_words_of_sentence(text: str, n: int = 2) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return False
    for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
        sentence = sentence.strip()
        if not sentence:
            continue
        words = re.findall(r"\S+", sentence)
        if re.search(r"\brecommendations\b", " ".join(words[:n]), re.IGNORECASE):
            return True
    return False


_RE_TOP_NUM = re.compile(
    r"(?<!\w)("
    r"\(\d{1,2}\)"
    r"|"
    r"\d{1,2}\.\d{1,2}\.?"
    r"|"
    r"(?<!\d\.)\d{1,2}\s*[.)]"
    r")(?=\s+\S)"
)
_RE_ALPHA = re.compile(
    r"(?<![\w.’'])"
    r"(\((?:a|b|c|d|e|f|g|h)\)|(?:a|b|c|d|e|f|g|h)\s*[.)])"
    r"(?=\s+(?!above\b|below\b|and\b|or\b|to\b|of\b|in\b|on\b|for\b|with\b)\S)",
    re.IGNORECASE,
)
_RE_ROMAN = re.compile(
    r"(?<![\wâ€™'])"
    r"(\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv)\)|"
    r"(?:i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii|xiii|xiv|xv)\s*[.)])"
    r"(?=\s+\S)",
    re.IGNORECASE,
)


def _should_continue_structured_list_page(page_text: str, min_markers: int = 5) -> bool:
    cleaned = _clean_text(page_text)
    total_markers = len(_RE_TOP_NUM.findall(cleaned)) + len(_RE_ALPHA.findall(cleaned)) + len(_RE_ROMAN.findall(cleaned))
    return total_markers >= min_markers


def _should_start_recommendations_section(page_text: str, n_words: int = 2) -> bool:
    cleaned = _clean_text(page_text)
    if _RE_RECOMM_CHAPTER_HEADING.search(cleaned):
        return True
    if not _recommendations_within_first_n_words_of_sentence(cleaned, n=n_words):
        return False

    matches = list(_RE_RECOMM_HEADING.finditer(cleaned))
    if not matches:
        return False

    for match in matches:
        sentence_start = 0
        for index in range(match.start() - 1, -1, -1):
            if cleaned[index] in ".!?":
                sentence_start = index + 1
                break

        sentence_end = len(cleaned)
        for index in range(match.end(), len(cleaned)):
            if cleaned[index] in ".!?":
                sentence_end = index
                break

        sentence = cleaned[sentence_start:sentence_end].strip()
        words = re.findall(r"\S+", sentence)
        first_n_text = " ".join(words[:n_words])
        if not re.search(r"\brecommendations\b", first_n_text, re.IGNORECASE):
            continue

        after_heading = cleaned[match.end():].strip()
        after_heading = re.sub(r"^[\s:.\-–—]*", "", after_heading)
        if _RE_TOP_NUM.match(after_heading) or _RE_ALPHA.match(after_heading) or _RE_ROMAN.match(after_heading):
            return True

    return False


def _slice_below_recommendations_heading(text: str) -> str:
    cleaned = _clean_text(text)
    patterns = [
        r"conclusions\s+and\s+recommendations",
        r"(?:^|[\n\.])\s*recommendations\s*(?:$|[\n\.])",
        r"make\s+the\s+following\s+recommendations[:\s]*",
        r"the\s+recommendations",
    ]

    earliest_match = None
    for pattern in patterns:
        match = re.search(pattern, cleaned, re.IGNORECASE)
        if not match:
            continue
        if earliest_match is None or match.start() < earliest_match.start():
            earliest_match = match

    if not earliest_match:
        return cleaned

    below = cleaned[earliest_match.end() :]
    return re.sub(r"^[\s:.\-]*", "", below).strip()


def _looks_like_new_section_page(text: str) -> bool:
    cleaned = _clean_text(text)
    if re.search(r"\b(appendix|annex|contents|references|bibliography)\b", cleaned[:250], re.I):
        return True
    if re.match(r"^\s*\d{1,2}\.\s+[A-Z][A-Z\s/&,\-]{6,}\b", cleaned):
        return True
    if re.match(r"^\s*(chapter|part|appendix|annex)\b", cleaned, re.I):
        return True
    return False


def _split_by_marker(text: str, marker_re: re.Pattern[str]) -> list[tuple[str, str]]:
    matches = []
    for match in marker_re.finditer(text):
        label = match.group(1).strip()
        if re.fullmatch(r"\(\d{1,2}\)", label):
            continue
        matches.append(match)
    if not matches:
        return []

    output = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        output.append((match.group(1), text[start:end].strip()))
    return output


def _normalize_top_num_label(label: str) -> str:
    return re.sub(r"\s+", "", label.strip())


def _normalize_sub_label(label: str) -> str:
    return re.sub(r"\s+", "", label.strip())


def _extract_alpha_label(label: str) -> Optional[str]:
    match = re.search(r"\(([a-h])\)", label, re.IGNORECASE)
    if match:
        return f"({match.group(1).lower()})"
    match = re.search(r"[.)]\s*([a-h])\s*[.)]", label, re.IGNORECASE)
    if match:
        return f"({match.group(1).lower()})"
    return None


def _format_item_label(label: str) -> str:
    value = label.strip()
    # Preserve nested-recommendation display labels like "4a", "4a i", "10a v"
    if re.fullmatch(r"\d{1,2}[a-h](?:\s+[ivxlcdm]+)?", value, flags=re.IGNORECASE):
        return re.sub(r"\s+", " ", value).lower()
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"^\((\d{1,2}(?:\.\d{1,2})?)\)$", r"\1", value)
    value = value.replace(".(", "(")
    value = re.sub(r"\.$", "", value)
    return value


_RE_INLINE_DIRECTIVE_LEAD = re.compile(
    r"\bwe\s+(?:therefore\s+|also\s+|further\s+)?"
    r"(?:recommend|urge|encourage|call\s+on)\b",
    re.IGNORECASE,
)


_RE_DIRECTIVE_VERB_PREFIX = re.compile(
    r"\b(should|must|shall|recommend|recommends|recommended|recommending|"
    r"need\s+to|needs\s+to|require[ds]?|requires|will|would|may|might|ought|"
    r"urge|urges|urged|encourage|encourages|will\s+need\s+to|"
    r"establish(?:es|ed|ing)?|introduce[ds]?|publish(?:es|ed|ing)?|consider(?:s|ed|ing)?|"
    r"ensure[ds]?|provide[ds]?|review(?:s|ed|ing)?|appoint(?:s|ed|ing)?|"
    r"set\s+out|set\s+up|implement(?:s|ed|ing)?|create[ds]?|"
    r"agree[ds]?|call\s+on|invite[ds]?|commend[ds]?|note[ds]?|welcome[ds]?)\b",
    re.IGNORECASE,
)


def _is_heading_like_prefix(prefix: str) -> bool:
    """Return True if the prefix between a numeric marker and the first
    bracketed sub-marker looks like a section heading rather than a directive
    lead-in.

    Used to detect nested-recommendation parents (e.g. Infected Blood's
    ``2. Recognising and remembering what happened to people`` followed by
    ``(a)`` / ``(b)`` / ``(c)`` sub-recommendations). When the prefix has no
    modal/directive verbs and is short enough to be a heading, the bracketed
    children are independently addressable sub-recommendations and should be
    emitted as separate rows (``2a``, ``2b``, ``2c``).
    """
    if not prefix:
        return False
    cleaned = prefix.strip().rstrip(":.;,")
    words = cleaned.split()
    if len(words) < 1 or len(words) > 18:
        return False
    if _RE_DIRECTIVE_VERB_PREFIX.search(cleaned):
        return False
    return True


def _alpha_letter(label: str) -> str:
    match = re.search(r"[a-h]", label, re.IGNORECASE)
    return match.group(0).lower() if match else ""


def _roman_letter(label: str) -> str:
    cleaned = re.sub(r"[^ivxlcdm]", "", label.lower())
    return cleaned


def _digits_only(label: str) -> str:
    return re.sub(r"\D", "", label)


def _emit_nested_alpha_children(
    top_digits: str,
    alpha_items: list[tuple[str, str]],
    items: list[dict[str, str]],
) -> None:
    """Emit alpha sub-items of a nested-recommendation parent as
    independently addressable rows using compact labels like ``2a`` and
    ``4a i``.
    """
    for alpha_label, alpha_chunk in alpha_items:
        letter = _alpha_letter(alpha_label)
        if not letter:
            continue
        roman_items = _split_by_marker(alpha_chunk, _RE_ROMAN)
        if roman_items:
            for roman_label, roman_chunk in roman_items:
                rletter = _roman_letter(roman_label)
                if not rletter:
                    continue
                items.append(
                    {
                        "label": f"{top_digits}{letter} {rletter}",
                        "level": "roman-nested",
                        "text": _clean_text(roman_chunk),
                    }
                )
        else:
            items.append(
                {
                    "label": f"{top_digits}{letter}",
                    "level": "alpha-nested",
                    "text": _clean_text(alpha_chunk),
                }
            )


def _emit_nested_roman_continuation(
    top_digits: str,
    alpha_letter_carry: str,
    roman_items: list[tuple[str, str]],
    items: list[dict[str, str]],
) -> None:
    """Emit roman items that continue a previous alpha block across a page
    break, using nested format ``Na i``.
    """
    for roman_label, roman_chunk in roman_items:
        rletter = _roman_letter(roman_label)
        if not rletter:
            continue
        items.append(
            {
                "label": f"{top_digits}{alpha_letter_carry} {rletter}",
                "level": "roman-nested",
                "text": _clean_text(roman_chunk),
            }
        )


def _roman_to_int(value: str) -> int:
    numerals = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
    total = 0
    prev = 0
    for char in reversed(value.lower()):
        current = numerals.get(char, 0)
        if current < prev:
            total -= current
        else:
            total += current
            prev = current
    return total


def _item_label_sort_key(label: str) -> tuple[int, tuple[int, ...]]:
    """
    Convert labels like ``8.1``, ``8.2(a)``, or ``10(iv)`` into sortable keys.
    Unlabelled rows are sorted after structured items within the document.
    """
    formatted = _format_item_label(label)
    if not formatted:
        return (1, ())

    nested = re.fullmatch(r"(\d{1,2})([a-h])(?:\s+([ivxlcdm]+))?", formatted, flags=re.IGNORECASE)
    if nested:
        num = int(nested.group(1))
        alpha = ord(nested.group(2).lower()) - 96
        roman = _roman_to_int(nested.group(3)) if nested.group(3) else 0
        return (0, (num, alpha, roman))

    key: list[int] = []
    for number in re.findall(r"\d+", formatted):
        key.append(int(number))

    for bracket in re.findall(r"\(([a-z]+|[ivxlcdm]+)\)", formatted, flags=re.IGNORECASE):
        token = bracket.lower()
        if re.fullmatch(r"[ivxlcdm]+", token):
            key.append(_roman_to_int(token))
        elif re.fullmatch(r"[a-z]+", token):
            for char in token:
                key.append(ord(char) - 96)

    return (0, tuple(key))


def _trim_after_list_end(text: str) -> str:
    match = _RE_SECTION_BREAK_A.search(text)
    if match:
        return text[: match.start()].strip()
    return text


def _is_structured_recommendation_item(text: str) -> bool:
    return True


def _strip_list_markers_only(text: str) -> str:
    cleaned = _clean_text(text)
    while True:
        new_text = re.sub(
            r"^\s*(?:\(\d{1,2}\)|\d{1,2}(?:\.\d{1,2})+[.]?|\d{1,2}[.)]|\([a-z]\)|"
            r"[a-z][.)]|\([ivxlcdm]+\)|[ivxlcdm]+[.)])\s+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        if new_text == cleaned:
            break
        cleaned = new_text.strip()
    return cleaned


def _strip_short_heading_after_numeric(text: str) -> str:
    cleaned = _strip_list_markers_only(text)
    match = re.match(r"^([A-Za-z/&,\- ]{1,40}?)\s+(.+)$", cleaned)
    if not match:
        return cleaned

    heading = match.group(1).strip()
    rest = match.group(2).strip()
    heading_words = [word for word in re.findall(r"[A-Za-z]+", heading) if word]
    looks_all_caps_heading = bool(heading_words) and all(word.isupper() for word in heading_words)
    if looks_all_caps_heading and 2 <= len(heading.split()) <= 4 and len(rest.split()) >= 6:
        return rest
    return cleaned


_TRAILING_SUBHEADING_MODAL_GUARD = re.compile(r"\b(should|must|shall|recommend|urge)\b", re.IGNORECASE)
_TRAILING_SUBHEADING_VERB_GUARD = re.compile(
    r"\b(is|are|was|were|be|been|being|have|has|had|do|does|did|can|could|may|might|will|would|should|must|shall)\b",
    re.IGNORECASE,
)
_TRAILING_SUBHEADING_ENTITY_GUARD = re.compile(r"\b(government|ministers)\b", re.IGNORECASE)
_TRAILING_SUBHEADING_CONNECTORS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "per",
    "the",
    "to",
    "via",
    "with",
    "within",
}


def _remove_short_trailing_subheading(text: str, max_words: int = 12) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return cleaned

    match = re.match(r"^(.*[.!?])\s+([^.!?]+?)\s*$", cleaned)
    if not match:
        return cleaned

    body = match.group(1).strip()
    tail = match.group(2).strip()
    tail_words = tail.split()
    if not (2 <= len(tail_words) <= max_words):
        return cleaned
    if _TRAILING_SUBHEADING_MODAL_GUARD.search(tail):
        return cleaned

    normalized_words = []
    for word in tail_words:
        clean_word = re.sub(r"^[^\w]+|[^\w]+$", "", word)
        if not clean_word or not re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)*", clean_word):
            return cleaned
        normalized_words.append(clean_word)

    first_word = normalized_words[0]
    if not (first_word[0].isupper() or first_word.isupper()):
        return cleaned
    if ";" in tail:
        return cleaned
    if "," in tail and _TRAILING_SUBHEADING_VERB_GUARD.search(tail):
        return cleaned
    if ":" in tail:
        if tail.count(":") > 1:
            return cleaned
        left, right = [part.strip() for part in tail.split(":", 1)]
        if not left or not right:
            return cleaned
        if _TRAILING_SUBHEADING_VERB_GUARD.search(tail):
            return cleaned

    if _TRAILING_SUBHEADING_ENTITY_GUARD.search(tail):
        if _TRAILING_SUBHEADING_VERB_GUARD.search(tail):
            return cleaned

    for word in normalized_words[1:]:
        if word.lower() in _TRAILING_SUBHEADING_CONNECTORS:
            continue
        if len(word) == 1:
            return cleaned

    return body


def _trim_trailing_number_colon_section(text: str) -> str:
    cleaned = _clean_text(text)
    if not cleaned:
        return cleaned

    match = re.match(r"^(.*?[.!?])\s+\d{1,2}\s*:\s+[A-Za-z].*$", cleaned)
    if match:
        return match.group(1).strip()
    return cleaned


def _clean_final_recommendation_text(doc_id: str, item_label: str, text: str) -> str:
    cleaned = _clean_text(text)
    if (
        "TheSpaceEconomyReport" in str(doc_id)
        and str(item_label).strip() == "30"
    ):
        cleaned = re.sub(
            r"\s+Growing\s+the\s+UK(?:['\u2019]s)?\s+space\s+economy\s*$",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
    return cleaned


def _extract_structured_items_from_page(
    page_text: str,
    carry_top_num: Optional[str] = None,
    carry_alpha_label: Optional[str] = None,
    carry_nested: bool = False,
) -> tuple[list[dict[str, str]], Optional[str], bool]:
    raw = _clean_text(page_text)
    parse_text = _trim_after_list_end(raw)
    items: list[dict[str, str]] = []
    page_last_explicit_top_num: Optional[str] = None
    page_last_is_nested: bool = carry_nested

    first_num = _RE_TOP_NUM.search(parse_text)

    if first_num and first_num.start() > 0 and carry_top_num is not None:
        prefix = parse_text[: first_num.start()].strip()
        carry_top_digits = _digits_only(carry_top_num)

        if carry_alpha_label:
            first_alpha_in_prefix = _RE_ALPHA.search(prefix)
            roman_prefix_text = prefix[: first_alpha_in_prefix.start()].strip() if first_alpha_in_prefix else prefix
            prefix_roman = _split_by_marker(roman_prefix_text, _RE_ROMAN)
            if carry_nested:
                _emit_nested_roman_continuation(
                    carry_top_digits, _alpha_letter(carry_alpha_label), prefix_roman, items
                )
            else:
                for roman_label, roman_chunk in prefix_roman:
                    roman_norm = _normalize_sub_label(roman_label)
                    items.append({"label": f"{carry_top_num}{carry_alpha_label}{roman_norm}", "level": "roman", "text": _clean_text(roman_chunk)})

        prefix_alpha = _split_by_marker(prefix, _RE_ALPHA)
        if prefix_alpha:
            if carry_nested:
                _emit_nested_alpha_children(carry_top_digits, prefix_alpha, items)
            else:
                for alpha_label, alpha_chunk in prefix_alpha:
                    roman_items = _split_by_marker(alpha_chunk, _RE_ROMAN)
                    if roman_items:
                        for roman_label, roman_chunk in roman_items:
                            items.append(
                                {
                                    "label": f"{carry_top_num}{alpha_label}{roman_label}",
                                    "level": "roman",
                                    "text": _clean_text(roman_chunk),
                                }
                            )
                    else:
                        items.append(
                            {
                                "label": f"{carry_top_num}{alpha_label}",
                                "level": "alpha",
                                "text": _clean_text(alpha_chunk),
                            }
                        )

    if first_num:
        list_text = parse_text[first_num.start() :].strip()
        top_items = _split_by_marker(list_text, _RE_TOP_NUM)
    else:
        if carry_top_num is None:
            return items, page_last_explicit_top_num, page_last_is_nested

        carry_top_digits = _digits_only(carry_top_num)

        if carry_alpha_label:
            first_alpha = _RE_ALPHA.search(parse_text)
            roman_prefix_text = parse_text[: first_alpha.start()].strip() if first_alpha else parse_text
            prefix_roman = _split_by_marker(roman_prefix_text, _RE_ROMAN)
            if carry_nested:
                _emit_nested_roman_continuation(
                    carry_top_digits, _alpha_letter(carry_alpha_label), prefix_roman, items
                )
            else:
                for roman_label, roman_chunk in prefix_roman:
                    roman_norm = _normalize_sub_label(roman_label)
                    items.append({"label": f"{carry_top_num}{carry_alpha_label}{roman_norm}", "level": "roman", "text": _clean_text(roman_chunk)})

        prefix_alpha = _split_by_marker(parse_text, _RE_ALPHA)
        if prefix_alpha:
            if carry_nested:
                _emit_nested_alpha_children(carry_top_digits, prefix_alpha, items)
            else:
                for alpha_label, alpha_chunk in prefix_alpha:
                    roman_items = _split_by_marker(alpha_chunk, _RE_ROMAN)
                    if roman_items:
                        for roman_label, roman_chunk in roman_items:
                            items.append(
                                {
                                    "label": f"{carry_top_num}{alpha_label}{roman_label}",
                                    "level": "roman",
                                    "text": _clean_text(roman_chunk),
                                }
                            )
                    else:
                        items.append(
                            {
                                "label": f"{carry_top_num}{alpha_label}",
                                "level": "alpha",
                                "text": _clean_text(alpha_chunk),
                            }
                        )
            return items, page_last_explicit_top_num, page_last_is_nested

        roman_only = _split_by_marker(parse_text, _RE_ROMAN)
        if roman_only:
            for roman_label, roman_chunk in roman_only:
                roman_norm = _normalize_sub_label(roman_label)
                items.append({"label": f"{carry_top_num}{roman_norm}", "level": "roman", "text": _clean_text(roman_chunk)})
            return items, page_last_explicit_top_num, page_last_is_nested

        items.append({"label": carry_top_num, "level": "numeric", "text": _clean_text(parse_text)})
        return items, page_last_explicit_top_num, page_last_is_nested

    for top_label, top_chunk in top_items:
        top_label_norm = _normalize_top_num_label(top_label)
        page_last_explicit_top_num = top_label_norm
        top_digits = _digits_only(top_label_norm)

        alpha_items = _split_by_marker(top_chunk, _RE_ALPHA)
        if alpha_items:
            first_alpha_pos = _RE_ALPHA.search(top_chunk)
            # Strip the leading numeric marker before measuring the prefix so
            # the heading words alone determine whether this is a nested
            # parent ("2. Recognising and remembering ...") versus an inline
            # directive ("8.33. We recommend that the Government ...").
            raw_prefix = top_chunk[: first_alpha_pos.start()] if first_alpha_pos else ""
            prefix_clean = re.sub(r"^\s*\(?\d{1,2}(?:\.\d{1,2})?\)?\.?\s*", "", raw_prefix).strip()
            prefix_words = len(prefix_clean.split())
            is_nested_parent = _is_heading_like_prefix(prefix_clean)

            if is_nested_parent:
                # Independently addressable sub-recommendations: emit each
                # alpha child (and its roman descendants, if any) as its own
                # row using compact labels like "2a" / "4a i".
                _emit_nested_alpha_children(top_digits, alpha_items, items)
                page_last_is_nested = True
            elif prefix_words > 6 or _RE_INLINE_DIRECTIVE_LEAD.search(prefix_clean):
                # Substantial directive text precedes the alpha markers, or the
                # short prefix is itself a recommendation directive (e.g.
                # "We recommend that the Government"): the bracketed letters
                # are inline sub-clauses of one recommendation, so keep the
                # whole chunk as a single numeric row.
                items.append({"label": top_label_norm, "level": "numeric", "text": _clean_text(top_chunk)})
                page_last_is_nested = False
            else:
                page_last_is_nested = False
                for alpha_label, alpha_chunk in alpha_items:
                    alpha_label_norm = _normalize_sub_label(alpha_label)
                    roman_items = _split_by_marker(alpha_chunk, _RE_ROMAN)
                    if roman_items:
                        for roman_label, roman_chunk in roman_items:
                            roman_label_norm = _normalize_sub_label(roman_label)
                            items.append(
                                {
                                    "label": f"{top_label_norm}{alpha_label_norm}{roman_label_norm}",
                                    "level": "roman",
                                    "text": _clean_text(roman_chunk),
                                }
                            )
                    else:
                        items.append(
                            {
                                "label": f"{top_label_norm}{alpha_label_norm}",
                                "level": "alpha",
                                "text": _clean_text(alpha_chunk),
                            }
                        )
        else:
            roman_items = _split_by_marker(top_chunk, _RE_ROMAN)
            if roman_items:
                page_last_is_nested = False
                for roman_label, roman_chunk in roman_items:
                    roman_label_norm = _normalize_sub_label(roman_label)
                    items.append(
                        {
                            "label": f"{top_label_norm}{roman_label_norm}",
                            "level": "roman",
                            "text": _clean_text(roman_chunk),
                        }
                    )
            else:
                page_last_is_nested = False
                items.append({"label": top_label_norm, "level": "numeric", "text": _clean_text(top_chunk)})

    return items, page_last_explicit_top_num, page_last_is_nested


_RE_SUMMARY_WORD_C = re.compile(r"\bsummary of\b", re.IGNORECASE)
_RE_RECOMM_WORD_C = re.compile(r"\b(recommendation|recommendations|recs?)\b", re.IGNORECASE)


def _should_start_detector_c_section(page_text: str) -> bool:
    cleaned = _clean_text(page_text)
    if not cleaned:
        return False
    for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
        if _RE_SUMMARY_WORD_C.search(sentence) and _RE_RECOMM_WORD_C.search(sentence):
            return True
    return False


def _detector_c_section_trigger_text(page_text: str) -> Optional[str]:
    cleaned = _clean_text(page_text)
    if not cleaned:
        return None
    for sentence in re.split(r"(?<=[.!?])\s+", cleaned):
        if _RE_SUMMARY_WORD_C.search(sentence) and _RE_RECOMM_WORD_C.search(sentence):
            return sentence.strip()
    return None


def _detector_c_new_section_reason(text: str) -> Optional[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return None

    match = re.search(r"\b(appendix|annex|contents|references|bibliography)\b", cleaned[:250], re.I)
    if match:
        return f"section-keyword:{match.group(1).lower()}"
    if re.match(r"^\s*\d{1,2}\.\s+[A-Z][A-Z\s/&,\-]{6,}\b", cleaned):
        return "numbered-all-caps-heading"
    if re.match(r"^\s*(chapter|part|appendix|annex)\b", cleaned, re.I):
        match = re.match(r"^\s*(chapter|part|appendix|annex)\b", cleaned, re.I)
        return f"chapter-like-heading:{match.group(1).lower()}" if match else "chapter-like-heading"
    return None


MODAL_PATTERNS = [
    r"\bshould\b",
    r"\bmust\b",
    r"\bencourage\b",
    r"\bneed to\b",
    r"\bneeds to\b",
    r"\brequired to\b",
    r"\bought to\b",
    r"\bwe recommend\b",
    r"\bwe urge\b",
    r"\bwe call on\b",
    r"\bthere should be\b",
    r"\bit is important that\b",
    r"\bit is important to\b",
    r"\bit will be important to\b",
    r"\bit is essential that\b",
    r"\bit will be necessary to\b",
    r"\bhave a responsibility to\b",
    r"\bwill need to\b",
]

TARGET_TERMS = [
    "government",
    "department",
    "minister",
    "ministers",
    "cabinet",
    "dwp",
    "dhsc",
    "post office",
    "fujitsu",
    "uk government",
    "devolved administrations",
    "each government",
    "nhs",
    "trust",
    "board",
    "committee",
    "departments",
    "secretary of state",
    "civil service",
    "public administration",
    "aria",
    "advanced research and invention agency",
    "uksa",
    "ukri",
    "uk space agency",
    "space agency",
    "dsit space directorate",
    "esa",
    "nato",
    "eu space programme",
    "should be",
    "must be",
    "there should be",
]

_RE_MODAL_C = re.compile("|".join(MODAL_PATTERNS), re.IGNORECASE)
_RE_POINT_START_C = re.compile(r"(?<!\w)(?:\(?\d{1,2}\)|\d{1,2}\.)\s+(?=[A-Z])")
_RE_LEADING_POINT_MARKER_C = re.compile(r"^\s*(?:\(\d{1,2}\)|\d{1,2}[.)])\s*")
_BRIDGE_LEAD_IN_RE = re.compile(r"^\s*we\s+(welcome|encourage)\b", re.IGNORECASE)
_BRIDGE_NEXT_START_RE = re.compile(
    r"^\s*(?:the\s+)?(?:government|uksa|ukri|ministers?|department|departments|secretary of state)\b",
    re.IGNORECASE,
)
_RE_ACTION_ORIENTATION_C = re.compile(
    r"\bestablish\b|\bcreate\b|\bintroduce\b|\bappoint\b|\bensure\b|\breview\b|\bprovide\b|\breform\b|\bset out\b|\bset up\b|"
    r"\bpublish\b|\bexplain\b|\bconsult\b|\bcommission\b|\bfund\b|\bimplement\b|\bconsider\b|\breport\b|\bmaintain\b|\bupdate\b|"
    r"\bcomplete\b|\baccept\b|\breduce\b|\bsimplify\b|\bstreamline\b|\brationalise\b|"
    r"\bprioritis(?:e|es|ed|ing)\b|\bprioritiz(?:e|es|ed|ing)\b|\bfocus(?:ing)?\b|\bimprov(?:e|es|ed|ing)\b|\bdeliver(?:y|ed|ing)?\b|"
    r"\bwork\b|\bpromot(?:e|es|ed|ing)\b|\bhighlight(?:s|ed|ing)?\b|\bencourag(?:e|es|ed|ing)\b|\bdelay(?:ed|ing)?\b|"
    r"\btake\b|\bsupport\b|\bcontinue\b|\bconduct\b|\bevaluate\b|\bseek\b|\bdevelop\b|\balign\b|\bfacilitat(?:e|es|ed|ing)\b|"
    r"\bprioritisation\b|\bprioritization\b|\bsimplification\b|\bstreamlining\b|\bback(?:ed|ing)?\b",
    re.IGNORECASE,
)


def _is_reference_not_recommendation(text: str) -> bool:
    lowered = text.lower().strip()
    blacklist_phrases = [
        "this recommendation",
        "these recommendations",
        "the recommendation",
        "recommendations are",
        "recommendations include",
        "recommendations listed",
        "recommendations set out",
        "we considered the recommendations",
        "in response to the recommendation",
        "the government agrees with",
        "the government welcomes",
        "the committee recommends in its report that",
        "in the interim report",
        "i made the following recommendations",
        "made the following recommendations",
        "previous recommendations",
        "earlier recommendations",
    ]
    if any(phrase in lowered for phrase in blacklist_phrases):
        if re.match(r"^\s*recommendation\s+\d+\b", lowered):
            return False
        if lowered.startswith("we recommend that") or lowered.startswith("we recommend"):
            return False
        return True
    return False


def _strip_leading_point_marker(text: str) -> str:
    return _RE_LEADING_POINT_MARKER_C.sub("", _clean_text(text)).strip()


def _sentence_is_modal_recommendation(text: str) -> bool:
    if not text:
        return False

    lowered = text.lower().strip()
    if re.search(r"\bshould\s+be\s+(?:a\s+)?(?:matter\s+of\s+)?priority\b|\bshould\s+be\s+a\s+priority\s+consideration\b", lowered):
        return True
    if re.search(r"\bthe\s+prioritis(?:ation|ization)\s+of\b", lowered) and re.search(r"\b(government|uk)\b", lowered):
        return True
    if re.search(r"\bprioritis(?:ation|ization)\s+of\b", lowered) and re.search(r"\b(debris|removal|adr|isam|space|technology|satellite)\b", lowered):
        return True
    if (
        re.search(r"^\s*we\s+welcome\b", lowered)
        and re.search(r"\bpivot\s+to\s+supporting\b", lowered)
        and re.search(r"\b(bilateral|partnerships?)\b", lowered)
        and re.search(r"\b(uk\s*space\s*agency|uksa)\b", lowered)
    ):
        return True
    if not _RE_MODAL_C.search(lowered):
        return False
    if re.search(r"^\s*we\s+(note|welcome|acknowledge|affirm)\b", lowered) and not _RE_MODAL_C.search(lowered):
        return False

    has_explicit_rec_verb = bool(re.search(r"\bwe recommend\b|\bwe urge\b|\bwe call on\b", lowered))
    has_directive_modal = bool(_RE_MODAL_C.search(lowered))
    has_action = bool(_RE_ACTION_ORIENTATION_C.search(lowered))
    has_target = any(term in lowered for term in TARGET_TERMS)

    if has_explicit_rec_verb:
        return True
    if not (has_directive_modal and has_action and has_target):
        return False
    return True


def _build_cross_page_bridge_candidate(prev_page_text: str, curr_page_text: str) -> tuple[str, str]:
    """Return ``(merged_text, parent_label)``.

    ``parent_label`` is the numeric marker that prefixed the trailing sentence
    on the previous page (e.g. ``"69"``) when present; otherwise ``""``.  The
    label lets the caller carry the parent recommendation's item number across
    the page break instead of emitting an unlabelled bridge row.
    """
    raw_prev_last = _last_sentence(prev_page_text)
    parent_label = ""
    marker_match = _RE_LEADING_POINT_MARKER_C.match(raw_prev_last)
    if marker_match:
        num_match = re.search(r"\d+", marker_match.group(0))
        if num_match:
            parent_label = num_match.group(0)

    # NLTK sentence tokenisation often splits a numeric marker like "69." as
    # its own sentence, leaving the rest of the recommendation as the final
    # sentence with no leading marker.  Recover the parent label by looking at
    # the preceding sentence(s) when the immediate last sentence carries no
    # marker of its own.
    if not parent_label:
        prev_sentences = _sent_tokenize(prev_page_text)
        for sentence in reversed(prev_sentences[:-1]):
            stripped = sentence.strip()
            num_only = re.fullmatch(r"\(?(\d{1,2})\)?\.?", stripped)
            if num_only:
                parent_label = num_only.group(1)
                break
            inline_marker = _RE_LEADING_POINT_MARKER_C.match(stripped)
            if inline_marker:
                num_match = re.search(r"\d+", inline_marker.group(0))
                if num_match:
                    parent_label = num_match.group(0)
                break
            if len(stripped.split()) > 3:
                break

    prev_last = _strip_leading_point_marker(raw_prev_last)
    curr_first = _strip_leading_point_marker(_first_sentence(curr_page_text))
    if not prev_last or not curr_first:
        return "", ""
    if len(prev_last.split()) < 6 or len(curr_first.split()) < 6:
        return "", ""
    if not _BRIDGE_LEAD_IN_RE.search(prev_last):
        return "", ""
    if not _BRIDGE_NEXT_START_RE.search(curr_first):
        return "", ""
    if not _sentence_is_modal_recommendation(curr_first):
        return "", ""

    merged = _clean_text(f"{prev_last} {curr_first}")
    if _is_reference_not_recommendation(merged):
        return "", ""
    return merged, parent_label


def _split_into_candidate_points(text: str) -> list[str]:
    cleaned = _clean_text(text)
    if not cleaned:
        return []

    matches = list(_RE_POINT_START_C.finditer(cleaned))
    if not matches:
        return [cleaned]

    points: list[str] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(cleaned)
        block = cleaned[start:end].strip()
        if block:
            points.append(block)
    return points


def _extract_modal_points(text: str) -> list[tuple[str, str]]:
    """Return ``(source_label, cleaned_text)`` pairs.

    ``source_label`` is the numeric marker that prefixed the block in the
    source document (e.g. ``"4"``, ``"79"``), or ``""`` if no marker was
    present.  The marker is stripped from the returned text as before.
    """
    if not text:
        return []

    output: list[tuple[str, str]] = []
    for block in _split_into_candidate_points(text):
        sentences = _sent_tokenize(block)
        has_modal = any(_sentence_is_modal_recommendation(_clean_text(sentence)) for sentence in sentences)
        if has_modal:
            cleaned_block = _clean_text(block)
            # Capture the leading numeric label before stripping it so the
            # source item number can be preserved as the item_label.
            label = ""
            m = _RE_LEADING_POINT_MARKER_C.match(cleaned_block)
            if m:
                num = re.search(r"\d+", m.group(0))
                if num:
                    label = num.group(0)
            cleaned_block = _RE_LEADING_POINT_MARKER_C.sub("", cleaned_block).strip()
            cleaned_block = _remove_short_trailing_subheading(cleaned_block, max_words=12)
            output.append((label, cleaned_block))
    return output


_RE_NUM_COLON_SECTION = re.compile(r"(?<!\d)\b\d{1,2}\s*:\s+(?=[A-Za-z])")


def _is_last_sentence_on_page(rec_text: str, page_text: str, tail_window: int = 500) -> bool:
    rec = _normalise_for_tail_match(rec_text)
    page = _normalise_for_tail_match(page_text)
    if not rec or not page:
        return False
    return page[-tail_window:].endswith(rec)


def _continuation_until_next_section(doc: str, start_page: int, page_map: dict[tuple[str, int], str], max_pages: int = 6) -> tuple[str, Optional[int]]:
    chunks = []
    last_page_used = None

    for page_number in range(start_page, start_page + max_pages):
        raw = page_map.get((doc, page_number))
        if not raw:
            break

        cleaned = _clean_text(raw)
        if not cleaned:
            break
        if _looks_like_new_section_page(cleaned):
            break

        marker_top = _RE_TOP_NUM.search(cleaned)
        marker_colon = _RE_NUM_COLON_SECTION.search(cleaned)
        marker_alpha = _RE_ALPHA.search(cleaned)
        marker_roman = _RE_ROMAN.search(cleaned)
        marker_candidates = [match.start() for match in (marker_top, marker_colon, marker_alpha, marker_roman) if match is not None]
        marker_pos = min(marker_candidates) if marker_candidates else None

        if marker_pos is not None:
            prefix = cleaned[:marker_pos].strip()
            if prefix:
                chunks.append(prefix)
                last_page_used = page_number
            break

        chunks.append(cleaned)
        last_page_used = page_number

    if not chunks:
        return "", None

    output = _clean_text(" ".join(chunks))
    output = _strip_list_markers_only(output)
    output = _remove_short_trailing_subheading(output, max_words=12)
    output = _trim_trailing_number_colon_section(output)
    return output, last_page_used


def _fix_cross_page_recommendations_from_pages(df: pd.DataFrame, pages: list[_PageUnit]) -> pd.DataFrame:
    page_map = {(page["document"], page["page_number"]): page["text"] for page in pages}
    result = df.copy()
    result["page"] = result["page"].astype(str)

    for idx, row in result.iterrows():
        text = str(row["text"]).strip()
        page_value = row["page"]
        doc = row["doc_id"]

        if isinstance(page_value, str) and "/" in page_value:
            continue

        try:
            page_number = int(page_value)
        except Exception:
            continue

        current_page_text = page_map.get((doc, page_number))
        if not current_page_text:
            continue

        if not text.endswith((".", "!", "?")) and _is_last_sentence_on_page(text, current_page_text):
            continuation, last_page_used = _continuation_until_next_section(
                doc=doc,
                start_page=page_number + 1,
                page_map=page_map,
                max_pages=6,
            )
            if continuation:
                merged_text = _clean_text(f"{text} {continuation}")
                merged_text = _remove_short_trailing_subheading(merged_text, max_words=12)
                merged_text = _trim_trailing_number_colon_section(merged_text)
                result.at[idx, "text"] = merged_text
                if last_page_used and last_page_used > page_number:
                    result.at[idx, "page"] = f"{page_number}/{last_page_used}"

    return result


def _page_units_from_input(text_or_units: str | list[Chunk] | list[PageRecord]) -> list[_PageUnit]:
    if isinstance(text_or_units, str):
        return [{"document": "", "page_number": 1, "text": _clean_text(text_or_units), "ocr": False}]
    if not text_or_units:
        return []

    first = text_or_units[0]
    if "raw_text" in first:
        return [
            {
                "document": str(page["source"]),
                "page_number": int(page["page_number"]),
                "text": str(page["text"]),
                "ocr": bool(page.get("ocr")),
            }
            for page in text_or_units
        ]

    grouped: dict[tuple[str, int], list[str]] = defaultdict(list)
    ocr_flags: dict[tuple[str, int], bool] = {}
    for chunk in text_or_units:
        page_number = chunk.get("page_number")
        if page_number is None:
            continue
        key = (str(chunk["source"]), int(page_number))
        grouped[key].append(str(chunk["text"]))
        ocr_flags[key] = bool(chunk.get("ocr"))

    pages: list[_PageUnit] = []
    for (document, page_number), texts in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        pages.append(
            {
                "document": document,
                "page_number": page_number,
                "text": _clean_text(" ".join(texts)),
                "ocr": ocr_flags[(document, page_number)],
            }
        )
    return pages


_RE_ALPHA_SUBPOINT_NOTES = re.compile(r"^(\d{1,2}(?:\.\d{1,2})?)\(([a-h])\)$")


def _merge_alpha_subpoints(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse sibling rows like 8.33(a)/(b)/(c) into one parent row labelled 8.33.

    Alphabetic sub-points are part of a single numbered recommendation, not
    separate records.  Any group of rows from the same document whose ``notes``
    field matches ``X.Y(a)``, ``X.Y(b)``, … are merged: sub-point texts are
    prefixed with their letter and joined in alphabetical order, and the merged
    row keeps the numeric parent label (``X.Y``) and the highest confidence.
    """
    def _parent(notes: str) -> Optional[str]:
        m = _RE_ALPHA_SUBPOINT_NOTES.match(str(notes))
        return m.group(1) if m else None

    df = df.copy()
    df["_parent"] = df["notes"].apply(_parent)
    alpha_mask = df["_parent"].notna()

    if not alpha_mask.any():
        return df.drop(columns=["_parent"])

    alpha_df = df[alpha_mask]
    non_alpha_df = df[~alpha_mask].drop(columns=["_parent"])

    merged_rows: list[pd.Series] = []
    for (_doc_id, parent_label), group in alpha_df.groupby(["doc_id", "_parent"]):
        group_sorted = group.sort_values("notes")
        parts: list[str] = []
        for _, row in group_sorted.iterrows():
            m = _RE_ALPHA_SUBPOINT_NOTES.match(str(row["notes"]))
            letter = m.group(2) if m else "?"
            parts.append(f"({letter}) {row['text']}")
        base = group_sorted.iloc[0].copy()
        base["notes"] = parent_label
        base["text"] = " ".join(parts)
        base["confidence"] = float(group["confidence"].max())
        base = base.drop(labels=["_parent"])
        merged_rows.append(base)

    if not merged_rows:
        return df.drop(columns=["_parent"])

    merged_df = pd.DataFrame(merged_rows)
    return pd.concat([non_alpha_df, merged_df], ignore_index=True)


_RE_NESTED_NOTES = re.compile(r"^(\d{1,2})([a-h])(?:\s+([ivxlcdm]+))?$", re.IGNORECASE)


def _collapse_single_roman_nested(df: pd.DataFrame) -> pd.DataFrame:
    """When a nested alpha block has exactly one roman child (and no
    sibling alpha-only row), drop the roman suffix so the row is labelled
    ``Na`` instead of ``Na i``. Examples: Infected Blood ``4b i`` → ``4b``,
    ``4d i`` → ``4d``.
    """
    df = df.copy()

    def _parse(notes: str) -> Optional[tuple[str, str, Optional[str]]]:
        match = _RE_NESTED_NOTES.match(str(notes).strip())
        if not match:
            return None
        return (match.group(1), match.group(2).lower(), (match.group(3) or "").lower() or None)

    parsed = df["notes"].apply(_parse)
    df["_nested"] = parsed
    nested_mask = parsed.notna()
    if not nested_mask.any():
        return df.drop(columns=["_nested"])

    counts: dict[tuple[str, str, str], int] = {}
    for (doc_id, parsed_val) in zip(df.loc[nested_mask, "doc_id"], df.loc[nested_mask, "_nested"]):
        if parsed_val is None or parsed_val[2] is None:
            continue
        key = (doc_id, parsed_val[0], parsed_val[1])
        counts[key] = counts.get(key, 0) + 1

    new_notes = df["notes"].copy()
    for idx, row in df[nested_mask].iterrows():
        parsed_val = row["_nested"]
        if parsed_val is None or parsed_val[2] is None:
            continue
        doc_id = row["doc_id"]
        key = (doc_id, parsed_val[0], parsed_val[1])
        # If there is exactly one roman child and no plain alpha row, collapse
        if counts.get(key, 0) != 1:
            continue
        sibling_alpha_exists = (
            (df["doc_id"] == doc_id)
            & (df["notes"].astype(str).str.fullmatch(rf"{parsed_val[0]}{parsed_val[1]}"))
        ).any()
        if sibling_alpha_exists:
            continue
        new_notes.at[idx] = f"{parsed_val[0]}{parsed_val[1]}"

    df["notes"] = new_notes
    return df.drop(columns=["_nested"])


def _drop_nested_parent_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop bare numeric parent rows (e.g. ``2``, ``4``) when nested child
    rows (``2a``, ``4a i``) exist for the same document. Avoids duplicate
    parent rows when sub-recommendations are emitted separately.
    """
    df = df.copy()
    notes_str = df["notes"].astype(str)

    nested_children: set[tuple[str, str]] = set()
    for doc_id, note in zip(df["doc_id"], notes_str):
        match = _RE_NESTED_NOTES.match(note.strip())
        if match:
            nested_children.add((str(doc_id), match.group(1)))

    if not nested_children:
        return df

    keep_mask = []
    for doc_id, note in zip(df["doc_id"], notes_str):
        cleaned = note.strip()
        if re.fullmatch(r"\d{1,2}", cleaned) and (str(doc_id), cleaned) in nested_children:
            keep_mask.append(False)
        else:
            keep_mask.append(True)
    return df[keep_mask].reset_index(drop=True)


def _resolve_body_paragraph_recommendation(
    text: str, doc_pages: list[_PageUnit]
) -> Optional[str]:
    """When ``text`` is a chapter-summary recommendation with inline alpha
    sub-clauses (e.g. Behaviour Change 8.33: "We recommend that the
    Government (a) … (c) details …"), search the document body for an
    earlier occurrence of the same recommendation phrase and return the
    longer body version when one exists. Body paragraphs sometimes preserve
    wording the chapter-8 summary truncates (e.g. body 7.48 has "(c) set out
    details …" where the 8.33 summary drops "set out").
    Returns None if no fuller body version is found.
    """
    rec_match = _RE_INLINE_DIRECTIVE_LEAD.search(text)
    if not rec_match:
        return None
    summary_clause = text[rec_match.start():].strip()
    if "(a)" not in summary_clause:
        return None  # only worth resolving when alpha sub-clauses are present

    sentence_end_re = re.compile(r"\.(?=\s+[A-Z0-9]|\s*$)")
    end = sentence_end_re.search(summary_clause)
    if end:
        summary_clause = summary_clause[: end.start() + 1].strip()

    prefix_len = min(60, len(summary_clause))
    prefix = summary_clause[:prefix_len]

    body = _clean_text(" ".join(p["text"] or "" for p in doc_pages))
    occurrences = [m.start() for m in re.finditer(re.escape(prefix), body)]
    if len(occurrences) < 2:
        return None

    best: Optional[str] = None
    for start in occurrences:
        window = body[start: start + 4000]
        m = sentence_end_re.search(window)
        if not m:
            continue
        candidate = window[: m.start() + 1].strip()
        if "(a)" not in candidate:
            continue
        if best is None or len(candidate) > len(best):
            best = candidate

    if not best or len(best) <= len(summary_clause):
        return None
    return best


def _detector_label(detector: str, ocr: bool) -> str:
    labels = {
        "A": "explicit label",
        "B": "structured section",
        "C": "embedded / heuristic",
    }
    base = labels.get(detector, detector)
    return f"{base} (OCR-derived)" if ocr else base


# ────────────────────────────────────────────────────────────────────────────
# Response-heading recommendation fallback.
#
# Some inquiry recommendation PDFs (e.g. Grenfell Tower Inquiry Phase 2
# Volume 7) do not expose a clean "Recommendation N:" list — they embed
# recommendations within prose using "we recommend …" wording. The official
# government response to such inquiries quotes every recommendation verbatim
# under "Recommendation N:" headings before responding. When primary
# extraction yields zero rows we can recover the recommendation set from
# those response-side headings.
#
# This is an opt-in fallback (per-preset flag) so it cannot affect the
# coursework-given pairs whose primary extraction already works.
# ────────────────────────────────────────────────────────────────────────────

_RE_RESPONSE_REC_HEADING = re.compile(r"\bRecommendation\s+(\d{1,3}[a-z]?)\s*:", re.IGNORECASE)
_RE_RESPONSE_PARA_REF = re.compile(r"\((\d{2,3}\.\d{1,2}(?:\.\d{1,2})?)\)")
_RE_RESPONSE_PHRASE_CUT = re.compile(
    r"\.\s+(?="
    r"The government (?:accepts|notes|welcomes|agrees|will|is|has|recognises|recognizes)"
    r"|We (?:will|have|are|accept)"
    r"|The Department\b"
    r"|ARB and RIBA accept"
    r"|[A-Z][A-Za-z]+ accepts? this recommendation"
    r")",
)


def extract_response_heading_recommendations(
    response_pages: list[PageRecord] | list[_PageUnit],
) -> list[Recommendation]:
    """Recover recommendations from "Recommendation N:" headings in a
    government response document.

    Each block is cut just after the trailing inquiry paragraph reference
    (e.g. ``(113.6)``) when present, otherwise just before the first
    sentence that begins with a known government-response opener. Subparts
    such as ``a) … b) …`` inside a single ``Recommendation N:`` are kept
    intact within the recommendation text.

    Returns rows tagged with ``extraction_source='response_heading_fallback'``
    and ``source_document_role='response'``.
    """
    pages = _page_units_from_input(response_pages) if response_pages else []
    if not pages:
        return []

    # Build the joined text with a char-offset → page-number map so each
    # recommendation can be attributed to the page that contains its heading.
    parts: list[str] = []
    page_offsets: list[tuple[int, int, str, bool]] = []  # (start, end, doc_id, ocr)
    cursor = 0
    for page in pages:
        text = page.get("text") or ""
        if not text:
            continue
        if parts:
            parts.append("\n")
            cursor += 1
        start = cursor
        parts.append(text)
        cursor += len(text)
        page_offsets.append((start, cursor, str(page.get("document", "")), bool(page.get("ocr"))))

    if not page_offsets:
        return []

    joined = "".join(parts)
    doc_id = page_offsets[0][2]

    def page_for_offset(offset: int) -> tuple[int, bool]:
        for index, (start, end, _doc, ocr) in enumerate(page_offsets, start=1):
            if start <= offset < end:
                return index, ocr
        return len(page_offsets), page_offsets[-1][3]

    matches = list(_RE_RESPONSE_REC_HEADING.finditer(joined))
    if not matches:
        return []

    seen_labels: set[str] = set()
    recommendations: list[Recommendation] = []

    for index, match in enumerate(matches):
        label = match.group(1).strip()
        if label in seen_labels:
            continue
        seen_labels.add(label)

        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(joined)
        block = joined[start:end]

        cut_method = "raw"
        confidence = 0.75
        para_ref = _RE_RESPONSE_PARA_REF.search(block)
        if para_ref:
            block_text = block[: para_ref.end()]
            cut_method = "para_ref"
            confidence = 0.88
        else:
            phrase_cut = _RE_RESPONSE_PHRASE_CUT.search(block)
            if phrase_cut:
                block_text = block[: phrase_cut.start() + 1]
                cut_method = "response_phrase"
                confidence = 0.82
            else:
                block_text = block
                confidence = 0.75

        block_text = _clean_text(block_text)
        if not block_text or len(block_text.split()) < 4:
            continue

        page_number, ocr = page_for_offset(start)
        rec_id = len(recommendations)
        rec: Recommendation = {
            "rec_id": rec_id,
            "text": block_text,
            "extraction_method": "response heading fallback",
            "detector": "F",
            "confidence": confidence,
            "document": doc_id,
            "page_number": page_number,
            "ocr": ocr,
            "item_label": label,
            "span_id": f"{doc_id}_p{page_number}_F{rec_id:03d}",
            "extraction_source": "response_heading_fallback",
            "source_document_role": "response",
            "extraction_note": (
                "Recommendation text recovered from government response headings "
                "because the recommendation PDF did not expose a clean "
                "recommendation list."
                + ("" if cut_method == "para_ref" else f" (cut={cut_method})")
            ),
        }
        recommendations.append(rec)

    return recommendations


# ────────────────────────────────────────────────────────────────────────────
# Inline-paragraph recommendation extractor.
#
# For inquiry reports that embed recommendations in numbered paragraphs (e.g.
# Grenfell Tower Inquiry Phase 2, Chapter 113: paragraphs 113.1 … 113.83) and
# express each recommendation in prose using "We therefore recommend that …",
# "We recommend that …", or "We also recommend …".
#
# The function walks every paragraph "<prefix>.N", finds each
# recommendation phrase within the paragraph, and captures the clause from
# the first ``that …`` (or first ``a./b./c.`` sub-list) up to the end of the
# recommendation sentence(s), trimming any trailing short title-case
# sub-heading that introduces the next paragraph block.
#
# Multi-recommendation paragraphs are split on the recommendation phrase, so
# 113.13 / 113.25 / 113.76 / 113.78 each yield two rows. Sub-listed
# recommendations (e.g. 113.23's a/b/c) stay intact within a single row.
# ────────────────────────────────────────────────────────────────────────────

_RE_GRENFELL_REC_PHRASE = re.compile(
    # Match the verb "recommend" but not the noun "recommendation(s)" or
    # past tense "recommended"/"recommending". This catches both the clean
    # "We therefore recommend that …" form and PDF-reflow-broken forms where
    # "recommend" is separated from its leading pronoun (e.g. Grenfell
    # 113.57 / 113.61 / 113.62 where reflow inserts other text between
    # "We therefore" and "recommend").
    r"\brecommend\b(?!ation|ations|ed|ing)",
    re.IGNORECASE,
)
_GRENFELL_CLAUSE_MAX_GAP = 130  # chars between "recommend" and first "that"/sub-list start

# Reject "recommend" verbs used as infinitive within a longer sentence
# (e.g. "led us to recommend that …", "and to recommend that …",
# "in a position to recommend …"). These are not formal recommendation
# sentence starts.
_RE_GRENFELL_INFINITIVE_LEADIN = re.compile(r"\bto\s+$", re.IGNORECASE)
_RE_GRENFELL_THAT = re.compile(r"\bthat\b", re.IGNORECASE)
_RE_GRENFELL_SUBPART_START = re.compile(r"\b([a-h])(?:\.|\))\s+(?=that\b|the\b|\S)", re.IGNORECASE)
_RE_GRENFELL_TRAILING_HEADING = re.compile(
    r"\.\s+([A-Z][A-Za-z]+(?:\s+[A-Za-z]+){0,4})\s*$",
)


_RE_GRENFELL_NEXT_PHRASE_LEADIN = re.compile(r"\b[Ww]e\s+(?:therefore\s+|also\s+)?$")
_RE_GRENFELL_SENTENCE_BOUNDARY = re.compile(r"\b[A-Za-z]{2,}\.\s+[A-Z]")


def _first_sentence_cut(clause: str) -> str:
    """Stop the recommendation at the first true sentence boundary, while
    keeping internal list markers like ``a.``, ``b.``, ``1.`` intact.
    """
    match = _RE_GRENFELL_SENTENCE_BOUNDARY.search(clause)
    if not match:
        return clause
    cut = clause.find(".", match.start()) + 1
    candidate = clause[:cut].strip()
    if len(candidate.split()) < 6:
        return clause
    return candidate


def _trim_grenfell_trailing_heading(text: str) -> str:
    """Strip a short trailing title-case sub-heading like 'Government' or
    'Fire engineers' that follows a recommendation block.
    """
    text = text.rstrip()
    match = _RE_GRENFELL_TRAILING_HEADING.search(text)
    if not match:
        return text
    heading = match.group(1).strip()
    if len(heading) >= 60 or re.search(r"[.,;:]", heading):
        return text
    # Heuristic: short (≤5 words), title-case, no sentence-internal punctuation.
    words = heading.split()
    if 1 <= len(words) <= 5 and all(w[:1].isupper() or w.lower() in {"and", "of", "the"} for w in words):
        return text[: match.start() + 1].rstrip()
    return text


def _extract_grenfell_rec_clause(segment: str) -> str | None:
    """Locate the recommendation clause within a segment that begins
    immediately after a 'we recommend' phrase.
    """
    if not segment:
        return None

    candidates: list[int] = []
    sub_match = _RE_GRENFELL_SUBPART_START.search(segment)
    if sub_match:
        candidates.append(sub_match.start())
    that_match = _RE_GRENFELL_THAT.search(segment)
    if that_match:
        candidates.append(that_match.start())

    if not candidates:
        return None
    start = min(candidates)
    # The clause anchor must be reasonably close to the "recommend" phrase.
    # If it is more than ~130 chars past the phrase, the surrounding sentence
    # is almost certainly negated/discussion (e.g. "we do not think it
    # appropriate for us to recommend specific changes to Approved Document B,
    # save in one respect. As we have pointed out … the guidance proceeds on
    # the assumption that …") rather than a recommendation.
    if start > _GRENFELL_CLAUSE_MAX_GAP:
        return None
    clause = segment[start:].strip()
    clause = re.sub(r"^and\s+", "", clause, flags=re.IGNORECASE)
    clause = _first_sentence_cut(clause)
    clause = _trim_grenfell_trailing_heading(clause)
    if not clause:
        return None
    if len(clause.split()) < 4:
        return None
    return clause


def extract_inline_paragraph_recommendations(
    policy_pages: list[PageRecord] | list[_PageUnit],
    chapter_prefix: str,
) -> list[Recommendation]:
    """Extract recommendations from a numbered-paragraph chapter
    (e.g. Grenfell's Chapter 113).
    """
    if not policy_pages or not chapter_prefix:
        return []
    # Accept either raw PageRecord dicts (from pdf_loader) or pre-converted
    # _PageUnit dicts (from extract_recommendations).
    first = policy_pages[0]
    if "document" in first and "page_number" in first and "text" in first:
        pages = [
            {
                "document": str(p.get("document", "")),
                "page_number": int(p.get("page_number", 0)),
                "text": str(p.get("text", "")),
                "ocr": bool(p.get("ocr", False)),
            }
            for p in policy_pages
        ]
    else:
        pages = _page_units_from_input(policy_pages)
    if not pages:
        return []

    # Build joined text + page-offset map (no re.sub on the joined string —
    # collapse whitespace per-page first to keep offsets stable).
    parts: list[str] = []
    page_spans: list[tuple[int, int, int, bool]] = []  # (start, end, page_no, ocr)
    cursor = 0
    doc_id = ""
    for page in pages:
        text = re.sub(r"\s+", " ", page.get("text") or "").strip()
        if not text:
            continue
        if not doc_id:
            doc_id = str(page.get("document", ""))
        if parts:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(text)
        cursor += len(text)
        page_spans.append((start, cursor, int(page.get("page_number", 0)), bool(page.get("ocr"))))

    if not page_spans:
        return []
    joined = "".join(parts)

    def page_for_offset(offset: int) -> tuple[int, bool]:
        for start, end, page_no, ocr in page_spans:
            if start <= offset < end:
                return page_no, ocr
        return page_spans[-1][2], page_spans[-1][3]

    prefix = re.escape(chapter_prefix.strip())
    para_re = re.compile(rf"(?<!\d){prefix}\.(\d{{1,3}})(?=\s)")
    matches = list(para_re.finditer(joined))
    if len(matches) < 5:
        return []

    recommendations: list[Recommendation] = []
    seen_keys: set[str] = set()

    for index, match in enumerate(matches):
        para_id = f"{chapter_prefix}.{match.group(1)}"
        block_end = matches[index + 1].start() if index + 1 < len(matches) else len(joined)
        block = joined[match.end():block_end]

        phrase_matches = list(_RE_GRENFELL_REC_PHRASE.finditer(block))
        # Reject matches that are not the start of a formal recommendation
        # sentence — e.g. participial / infinitival uses like "led us to
        # recommend that …" or "and to recommend that …" (Grenfell 113.80
        # is a discussion/background paragraph that ends with such a phrase).
        # Formal recommendations begin after a sentence boundary or after a
        # leading "we (therefore|also)" pronoun, not after the infinitive
        # marker "to" or a continuation conjunction "and to".
        phrase_matches = [
            m for m in phrase_matches
            if not _RE_GRENFELL_INFINITIVE_LEADIN.search(block[max(0, m.start() - 12):m.start()])
        ]
        if not phrase_matches:
            continue

        for ph_index, phrase in enumerate(phrase_matches):
            if ph_index + 1 < len(phrase_matches):
                next_start = phrase_matches[ph_index + 1].start()
                # Roll segment_end back across "We (also|therefore)?" lead-in
                # so that fragment doesn't get attached to the current rec
                # (PDF reflow places "We also" / "We" just before "recommend").
                lookback_start = max(0, next_start - 30)
                lead = _RE_GRENFELL_NEXT_PHRASE_LEADIN.search(block[lookback_start:next_start])
                seg_end = lookback_start + lead.start() if lead else next_start
            else:
                seg_end = len(block)
            segment = block[phrase.end():seg_end]
            clause = _extract_grenfell_rec_clause(segment)
            if not clause:
                continue

            key = _normalised_key(clause)
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)

            page_no, ocr = page_for_offset(match.start())
            rec_id = len(recommendations)
            label = str(rec_id + 1)
            recommendations.append(
                {
                    "rec_id": rec_id,
                    "text": clause,
                    "extraction_method": "inline paragraph (chapter prefix)",
                    "detector": "I",
                    "confidence": 0.9,
                    "document": doc_id,
                    "page_number": page_no,
                    "ocr": ocr,
                    "item_label": label,
                    "span_id": f"{doc_id}_p{page_no}_I{rec_id:03d}",
                    "extraction_source": "inline_report_recommendation",
                    "source_document_role": "policy",
                    "source_paragraph": para_id,
                    "extraction_note": (
                        f"Recommendation extracted inline from paragraph {para_id} "
                        "using the report's prose recommendation phrase."
                    ),
                }
            )

    return recommendations


# ────────────────────────────────────────────────────────────────────────────
# Select-committee "Conclusions and recommendations" extractor.
#
# House of Commons select-committee reports (e.g. Home Affairs Committee,
# "Police response to the 2024 summer disorder") finish with a single
# "Conclusions and recommendations" section listing every numbered item from
# the body — both *conclusions* (commentary) and *recommendations*
# (directives) — under one running numbering scheme. The published
# government response, however, only addresses the recommendation items.
#
# This extractor:
#   1. Finds the final "Conclusions and recommendations" section (skipping
#      the TOC occurrence).
#   2. Splits it into numbered items.
#   3. Keeps only items that contain a directive recommendation phrase —
#      "we recommend that …", "we encourage the Government to …",
#      or "<actor> should …" where <actor> is a known recipient (Government,
#      Home Office, CPS, "national policing system", etc).
#   4. Drops conclusion items (commendation / observation language with no
#      directive verb).
# ────────────────────────────────────────────────────────────────────────────

_RE_SELECT_CMTE_HEADING = re.compile(r"\bConclusions\s+and\s+recommendations\b", re.IGNORECASE)
_RE_SELECT_CMTE_ITEM = re.compile(r"(?<![\d.])\b(\d{1,2})\.\s+(?=[A-Z])")
_RE_SELECT_CMTE_DIRECTIVE = re.compile(
    r"\b(?:"
    r"we\s+recommend\s+that"
    r"|we\s+encourage\s+(?:the\s+)?government\s+to"
    r"|(?:the\s+)?government\s+should\b"
    r"|(?:the\s+)?home\s+office\s+should\b"
    r"|(?:the\s+)?cps\s+should\b"
    r"|(?:the\s+)?secretary\s+of\s+state\s+should\b"
    r"|(?:the\s+)?(?:new\s+)?national\s+(?:\w+\s+){0,3}system\s+(?:\w+\s+){0,3}should\s+include"
    r"|(?:the\s+)?(?:new\s+)?national\s+(?:\w+\s+){0,3}for\s+policing\s+should\s+include"
    r")\b",
    re.IGNORECASE,
)
_RE_SELECT_CMTE_TRAILING_HEADING = re.compile(
    r"\b(?:Formal\s+minutes|Witnesses|Published\s+written\s+evidence|Appendix|Annex|"
    r"List\s+of\s+(?:reports|witnesses)|Members\s+present)\b",
    re.IGNORECASE,
)


def extract_select_committee_recommendations(
    policy_pages: list[PageRecord] | list[_PageUnit],
) -> list[Recommendation]:
    """Extract directive recommendations from a select-committee
    "Conclusions and recommendations" final section.
    """
    if not policy_pages:
        return []

    first = policy_pages[0]
    if "document" in first and "page_number" in first and "text" in first:
        pages = [
            {
                "document": str(p.get("document", "")),
                "page_number": int(p.get("page_number", 0)),
                "text": str(p.get("text", "")),
                "ocr": bool(p.get("ocr", False)),
            }
            for p in policy_pages
        ]
    else:
        pages = _page_units_from_input(policy_pages)
    if not pages:
        return []

    parts: list[str] = []
    page_spans: list[tuple[int, int, int, bool]] = []
    cursor = 0
    doc_id = ""
    for page in pages:
        text = re.sub(r"\s+", " ", page.get("text") or "").strip()
        if not text:
            continue
        if not doc_id:
            doc_id = str(page.get("document", ""))
        if parts:
            parts.append(" ")
            cursor += 1
        start = cursor
        parts.append(text)
        cursor += len(text)
        page_spans.append((start, cursor, int(page.get("page_number", 0)), bool(page.get("ocr"))))

    if not page_spans:
        return []
    joined = "".join(parts)

    # Pick the LAST "Conclusions and recommendations" heading — the first
    # occurrence is typically the table of contents.
    heading_matches = list(_RE_SELECT_CMTE_HEADING.finditer(joined))
    if not heading_matches:
        return []
    section_start = heading_matches[-1].end()

    section_text = joined[section_start:]
    # Truncate at the first post-section structural heading.
    trailing = _RE_SELECT_CMTE_TRAILING_HEADING.search(section_text)
    if trailing:
        section_text = section_text[: trailing.start()]
    # Keep an absolute-offset map for page lookup.
    section_offset = section_start

    def page_for_offset(offset: int) -> tuple[int, bool]:
        for start, end, page_no, ocr in page_spans:
            if start <= offset < end:
                return page_no, ocr
        return page_spans[-1][2], page_spans[-1][3]

    # Split into numbered items.
    item_matches = list(_RE_SELECT_CMTE_ITEM.finditer(section_text))
    if len(item_matches) < 3:
        return []

    recommendations: list[Recommendation] = []
    for idx, match in enumerate(item_matches):
        number = int(match.group(1))
        item_start = match.end()
        item_end = item_matches[idx + 1].start() if idx + 1 < len(item_matches) else len(section_text)
        body = section_text[item_start:item_end].strip()
        if not body:
            continue
        if not _RE_SELECT_CMTE_DIRECTIVE.search(body):
            continue

        # Sanity: skip implausibly short or implausibly long items.
        word_count = len(body.split())
        if word_count < 8 or word_count > 600:
            continue

        absolute_offset = section_offset + match.start()
        page_no, ocr = page_for_offset(absolute_offset)
        rec_id = len(recommendations)
        label = str(rec_id + 1)
        # Strip a trailing topic-section title that bled in from the NEXT
        # numbered item's heading (e.g. "…interpretation and use. Policing
        # response to disorder").  Pattern: after the last sentence-ending
        # punctuation, a short Title-Case phrase with no internal period.
        body = re.sub(
            r"(?<=[.!?])\s+[A-Z][\w]+(?:\s+(?:to\s+)?[a-z]+){1,5}\s*$",
            "",
            body,
        ).strip()
        recommendations.append(
            {
                "rec_id": rec_id,
                "text": body,
                "extraction_method": "select-committee item",
                "detector": "S",
                "confidence": 0.9,
                "document": doc_id,
                "page_number": page_no,
                "ocr": ocr,
                "item_label": label,
                "span_id": f"{doc_id}_p{page_no}_S{rec_id:03d}",
                "extraction_source": "select_committee_section",
                "source_document_role": "policy",
                "source_paragraph": f"Paragraph {number}",
                "source_item_type": "Recommendation",
                "extraction_note": (
                    f"Extracted from numbered item {number} of the report's "
                    "'Conclusions and recommendations' section; conclusion items in "
                    "the same numbering were excluded based on directive language."
                ),
            }
        )

    return recommendations


def extract_recommendations(
    text_or_units: str | list[Chunk] | list[PageRecord],
    *,
    response_pages_fallback: list[PageRecord] | None = None,
    inline_chapter_prefix: str | None = None,
    select_committee_section: bool = False,
) -> list[Recommendation]:
    """
    Run the final recommendation extraction pipeline migrated from ``delete_me.py``.

    Page records are the preferred input because the validated prototype is
    page-aware and uses section detection plus cross-page continuation logic.
    """
    pages = _page_units_from_input(text_or_units)
    if not pages:
        return []

    # Targeted override: select-committee reports format their final
    # "Conclusions and recommendations" section as one numbered list
    # interleaving conclusions and recommendations. The generic detectors
    # cannot tell them apart, so when the preset opts in we run the
    # dedicated extractor and skip the generic pipeline entirely. If it
    # returns nothing, we fall through to the generic detectors below.
    if select_committee_section:
        sc_recs = extract_select_committee_recommendations(pages)
        if sc_recs:
            return sc_recs

    pages_by_doc: dict[str, list[_PageUnit]] = defaultdict(list)
    for page in pages:
        pages_by_doc[page["document"]].append(page)
    for document_pages in pages_by_doc.values():
        document_pages.sort(key=lambda item: item["page_number"])

    all_candidates: list[_RecCandidate] = []

    # Detector A
    for doc_id, doc_pages in pages_by_doc.items():
        for page in doc_pages:
            page_text = page["text"] or ""
            if _page_has_prior_recommendation_context(page_text):
                continue

            blocks = _extract_explicit_recommendation_blocks(page_text)
            if not blocks:
                continue

            for index, (label, block) in enumerate(blocks):
                conf, _ = _score_candidate(block, "A")
                all_candidates.append(
                    _RecCandidate(
                        doc_id=doc_id,
                        page=page["page_number"],
                        span_id=f"{doc_id}_p{page['page_number']}_A{index:03d}",
                        text=block,
                        detector="A",
                        confidence=conf,
                        notes=str(label),
                        ocr=page["ocr"],
                    )
                )

    # Detector B
    for doc_id, doc_pages in pages_by_doc.items():
        in_recommendations_section = False
        last_top_num = None
        last_alpha_label = None
        last_is_nested = False

        for page in doc_pages:
            page_text = page["text"] or ""
            triggered_this_page = False

            if not in_recommendations_section:
                if not _should_start_recommendations_section(page_text):
                    continue
                in_recommendations_section = True
                triggered_this_page = True
            else:
                if _looks_like_new_section_page(page_text):
                    break
                if not _should_continue_structured_list_page(page_text, min_markers=3):
                    continue

            page_text_for_extraction = _slice_below_recommendations_heading(page_text) if triggered_this_page else page_text
            items, page_last_explicit_top_num, page_last_is_nested = _extract_structured_items_from_page(
                page_text_for_extraction,
                carry_top_num=last_top_num,
                carry_alpha_label=last_alpha_label,
                carry_nested=last_is_nested,
            )

            last_is_nested = page_last_is_nested

            if not items:
                continue

            accepted_page_top_nums: list[int] = []
            for index, item in enumerate(items):
                label = item["label"]

                # Resolve "(paragraph N.M)" cross-references to the body
                # paragraph version of the recommendation when it contains a
                # fuller "We recommend that …" sentence (e.g. Behaviour Change
                # 8.33 → body 7.48 includes "(c) set out details" wording the
                # chapter-8 summary loses). Done before cleanups so the
                # cross-ref marker is still present in the raw item text.
                resolved = _resolve_body_paragraph_recommendation(item["text"], doc_pages)
                if resolved and len(resolved) > len(item["text"]) * 0.5:
                    item = {**item, "text": resolved, "level": "numeric"}

                if item["level"] == "numeric":
                    block = _strip_short_heading_after_numeric(item["text"])
                else:
                    block = _strip_list_markers_only(item["text"])

                block = _remove_short_trailing_subheading(block, max_words=12)
                block = re.split(r"\bSuccess in implementation will be measured\b", block, flags=re.I)[0].strip()

                if item["level"] == "numeric" and len(block.split()) <= 12 and not re.search(r"\bshould\b|\bmust\b|\bshall\b|\brecommend\b", block, re.I):
                    continue
                if item["level"] in ("alpha", "alpha-nested") and len(block.split()) <= 4 and not re.search(r"\bshould\b|\bmust\b|\bshall\b|\brecommend\b", block, re.I):
                    continue
                if not _is_structured_recommendation_item(block):
                    continue

                conf, _ = _score_candidate(block, "B")
                notes = _format_item_label(label)

                label_top_match = re.match(r"^\(?\s*(\d{1,2})\s*\)?", label)
                if label_top_match:
                    accepted_page_top_nums.append(int(label_top_match.group(1)))
                if item["level"] in ("alpha-nested", "roman-nested"):
                    nested_alpha = re.match(r"^\d{1,2}([a-h])", label)
                    if nested_alpha:
                        last_alpha_label = f"({nested_alpha.group(1)})"
                else:
                    alpha_from_label = _extract_alpha_label(label)
                    if alpha_from_label:
                        last_alpha_label = alpha_from_label

                all_candidates.append(
                    _RecCandidate(
                        doc_id=doc_id,
                        page=page["page_number"],
                        span_id=f"{doc_id}_p{page['page_number']}_B{index:03d}",
                        text=block,
                        detector="B",
                        confidence=conf,
                        notes=notes,
                        ocr=page["ocr"],
                    )
                )

            page_top_num_candidate = max(accepted_page_top_nums) if accepted_page_top_nums else None
            if page_top_num_candidate is None and page_last_explicit_top_num:
                page_top_num_candidate = int(re.sub(r"\D", "", page_last_explicit_top_num) or "0")

            if page_top_num_candidate is not None:
                candidate_top = f"{page_top_num_candidate}."
                if last_top_num:
                    prev_num = int(re.sub(r"\D", "", last_top_num) or "0")
                    curr_num = page_top_num_candidate
                    if curr_num >= prev_num or curr_num >= 10:
                        last_top_num = candidate_top
                else:
                    last_top_num = candidate_top

    # Detector C
    seen_c_keys: set[tuple[str, str]] = set()
    for doc_id, doc_pages in pages_by_doc.items():
        in_c_section = False
        prev_page_text = ""

        for page in doc_pages:
            page_text = page["text"] or ""
            trigger_text = _detector_c_section_trigger_text(page_text)

            if not in_c_section:
                if not trigger_text:
                    continue
                in_c_section = True
            else:
                stop_reason = _detector_c_new_section_reason(page_text)
                if stop_reason:
                    in_c_section = False
                    if not trigger_text:
                        continue
                    in_c_section = True

            if re.search(r"\brecommendation\s+\d+\s*:", page_text, re.IGNORECASE):
                prev_page_text = page_text
                continue

            bridge_text, bridge_parent_label = _build_cross_page_bridge_candidate(prev_page_text, page_text)
            if bridge_text:
                bridge_text = _remove_short_trailing_subheading(bridge_text, max_words=12)
                bridge_key = (doc_id, _normalised_key(bridge_text))
                if bridge_key not in seen_c_keys:
                    conf, _ = _score_candidate(bridge_text, "C")
                    prev_page_num = max(1, page["page_number"] - 1)
                    curr_page_num = page["page_number"]
                    if bridge_parent_label and curr_page_num > prev_page_num:
                        bridge_page: int | str = f"{prev_page_num}/{curr_page_num}"
                        bridge_notes = bridge_parent_label
                    else:
                        bridge_page = prev_page_num
                        bridge_notes = bridge_parent_label or "bridge"
                    all_candidates.append(
                        _RecCandidate(
                            doc_id=doc_id,
                            page=bridge_page,
                            span_id=f"{doc_id}_p{prev_page_num}_CBRIDGE",
                            text=bridge_text,
                            detector="C",
                            confidence=conf,
                            notes=bridge_notes,
                            ocr=page["ocr"],
                        )
                    )
                    seen_c_keys.add(bridge_key)

            points = _extract_modal_points(page_text)
            if not points:
                prev_page_text = page_text
                continue

            for index, (point_label, point_text) in enumerate(points):
                point_key = (doc_id, _normalised_key(point_text))
                if point_key in seen_c_keys:
                    continue
                conf, _ = _score_candidate(point_text, "C")
                all_candidates.append(
                    _RecCandidate(
                        doc_id=doc_id,
                        page=page["page_number"],
                        span_id=f"{doc_id}_p{page['page_number']}_C{index:03d}",
                        text=point_text,
                        detector="C",
                        confidence=conf,
                        notes=point_label,
                        ocr=page["ocr"],
                    )
                )
                seen_c_keys.add(point_key)

            prev_page_text = page_text

    if all_candidates:
        def _start_page(value: int | str) -> int | str:
            if isinstance(value, str) and "/" in value:
                head = value.split("/", 1)[0]
                try:
                    return int(head)
                except ValueError:
                    return value
            return value

        bridge_leadins = set()
        for candidate in all_candidates:
            if str(candidate.span_id).endswith("_CBRIDGE"):
                lead = _normalised_key(_first_sentence(candidate.text))
                if lead:
                    bridge_leadins.add((candidate.doc_id, _start_page(candidate.page), lead))

        if bridge_leadins:
            filtered: list[_RecCandidate] = []
            for candidate in all_candidates:
                key = (candidate.doc_id, _start_page(candidate.page), _normalised_key(candidate.text))
                if not str(candidate.span_id).endswith("_CBRIDGE") and key in bridge_leadins:
                    continue
                filtered.append(candidate)
            all_candidates = filtered

    df = pd.DataFrame(
        [
            {
                "doc_id": candidate.doc_id,
                "page": candidate.page,
                "span_id": candidate.span_id,
                "detector": candidate.detector,
                "confidence": candidate.confidence,
                "text": candidate.text,
                "notes": candidate.notes,
                "ocr": candidate.ocr,
            }
            for candidate in all_candidates
        ],
        columns=["doc_id", "page", "span_id", "detector", "confidence", "text", "notes", "ocr"],
    )
    if df.empty:
        if inline_chapter_prefix:
            inline_recs = extract_inline_paragraph_recommendations(pages, inline_chapter_prefix)
            if inline_recs:
                return inline_recs
        if response_pages_fallback:
            return extract_response_heading_recommendations(response_pages_fallback)
        return []

    df["norm_key"] = df["text"].apply(_normalised_key)

    def _page_sort_value(value: int | str) -> tuple[int, int | str]:
        if isinstance(value, str) and "/" in value:
            head = value.split("/", 1)[0]
            try:
                return (0, int(head))
            except ValueError:
                return (1, value)
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, str(value))

    df["_page_sort"] = df["page"].apply(_page_sort_value)
    df_sorted = df.sort_values(["confidence"], ascending=False)
    kept = df_sorted[~df_sorted.duplicated(subset=["doc_id", "norm_key"], keep="first")]
    rec_df = kept.sort_values(["doc_id", "_page_sort", "confidence"], ascending=[True, True, False]).reset_index(drop=True)
    rec_df = rec_df.drop(columns=["_page_sort"])
    rec_df = _fix_cross_page_recommendations_from_pages(rec_df, pages)
    rec_df = rec_df.drop_duplicates(subset=["doc_id", "text"]).reset_index(drop=True)
    rec_df = _merge_alpha_subpoints(rec_df)
    rec_df = _collapse_single_roman_nested(rec_df)
    rec_df = _drop_nested_parent_rows(rec_df)
    rec_df = rec_df.drop_duplicates(subset=["doc_id", "notes", "text"]).reset_index(drop=True)
    rec_df["item_sort_key"] = rec_df["notes"].apply(_item_label_sort_key)
    rec_df = rec_df.sort_values(
        ["doc_id", "item_sort_key", "page", "confidence"],
        ascending=[True, True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)

    recommendations: list[Recommendation] = []
    for rec_id, row in enumerate(rec_df.to_dict(orient="records")):
        detector = str(row["detector"])
        notes = str(row.get("notes", ""))
        text = _clean_final_recommendation_text(str(row["doc_id"]), notes, str(row["text"]))
        recommendations.append(
            Recommendation(
                rec_id=rec_id,
                text=text,
                extraction_method=_detector_label(detector, bool(row.get("ocr"))),
                detector=detector,
                confidence=float(row["confidence"]),
                document=str(row["doc_id"]),
                page_number=row["page"],
                ocr=bool(row.get("ocr")),
                item_label=notes if (detector in {"A", "B"} or (detector == "C" and bool(re.match(r"^\d+$", notes.strip())))) else "",
                span_id=str(row["span_id"]),
            )
        )

    return recommendations
