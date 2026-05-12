"""Generate and validate the combined COP509 recommendation export.

The validator treats the supplied JSON export as the golden baseline and only
permits the row/field changes explicitly listed in the final tuning prompt.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.presets import PRESETS  # noqa: E402
from src.alignment import match_recommendations_to_response_units  # noqa: E402
from src.classification import classify_with_confidence, normalize_label  # noqa: E402
from src.extraction import extract_recommendations  # noqa: E402
from src.pdf_loader import extract_pages  # noqa: E402
from src.response_units import extract_response_units  # noqa: E402


DEBUG_FIELDS = [
    "alignment_confidence",
    "alignment_method",
    "classification_confidence",
    "classifier_method",
    "confidence_factors",
]

COMPARE_FIELDS = [
    "recommendation_page",
    "recommendation_text",
    "matched_response_page",
    "matched_response_text",
    "classification",
    "confidence",
    "debug.alignment_method",
    "debug.alignment_confidence",
    "debug.classification_confidence",
    "debug.classifier_method",
    "debug.confidence_factors",
]

EXPECTED_CLASSIFICATIONS = {
    ("covid_inquiry", "2"): "partial",
    ("covid_inquiry_module2", "1"): "not_addressed",
    ("covid_inquiry_module2", "13"): "not_addressed",
}

RESPONSE_TEXT_ALLOWED = {
    ("covid_inquiry_module2", value)
    for value in ["1", "2", "3", "5", "6", "9", "10", "13"]
} | {
    ("grenfell_phase2", value)
    for value in ["31", "32", "33", "34", "35", "36", "38", "39", "40", "41", "56"]
} | {
    ("space_economy", "23"),
    ("post_office", "4"),
    ("post_office", "18"),
    ("space_economy", "69"),
} | {
    ("blood_inquiry", value)
    for value in ["4a i", "4a ii", "4a iii", "4b", "4c i", "4c ii", "7a i", "7a ii", "7a iii"]
}

RECOMMENDATION_TEXT_ALLOWED = {("space_economy", "30")}
CLASSIFICATION_ALLOWED = set(EXPECTED_CLASSIFICATIONS)
CONFIDENCE_ALLOWED = RESPONSE_TEXT_ALLOWED | RECOMMENDATION_TEXT_ALLOWED | CLASSIFICATION_ALLOWED

LEAK_START_PATTERNS = [
    r"^:\s*That\b",
    r"^:\s*Chief\s+Medical\s+Officer\b",
    r"^:\s*Attendance\b",
    r"^:\s*Register\s+of\s+experts\b",
    r"^:\s*Support\s+to\s+participants\b",
    r"^:\s*Implementing\s+a\s+socio-economic\s+duty\b",
    r"^:\s*Delegated\s+powers\b",
    r"^:\s*Civil\s+emergency\s+decision-making\s+structures\b",
    r"^:\s*Amendment\s+of\s+the\s+Ministerial\s+Code\b",
    r"^Partnerships\s+in\s+the\s+aerospace\s+sector\b",
]

BLOOD_LEAK_PATTERNS = [
    r"\b4a\)\s*ii\.\s+The\s+operation\b",
    r"\b4a\)\s*iii\.\s+The\s+review\b",
    r"\b4c\)\s*i\.\s+That\s+external\s+regulation\b",
    r"\b4c\)\s*ii\.\s+That\s+the\s+national\s+healthcare\b",
    r"\b7a\)\s*ii\.\s+In\s+Scotland\b",
    r"\b7a\)\s*iii\.\s+Consideration\s+be\s+given\b",
]

LABEL_MAP = {
    "accepted": "accepted",
    "partially_accepted": "partial",
    "partial": "partial",
    "rejected": "rejected",
    "not_addressed": "not_addressed",
}

MISSING = object()


def _fixed(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return float(f"{value:.{digits}f}")


def _row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("document_pair") or ""), str(row.get("id") or "")


def _load_export(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("recommendations", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a recommendation list")
    return rows


def _write_export(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "total": len(rows),
        "recommendations": rows,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _overall_confidence(match: dict[str, Any] | None, classification: str | None, sim: float) -> tuple[float, list[str]]:
    if not match or not match.get("matched_text"):
        return 0.0, []

    alignment_confidence = float(match.get("alignment_confidence", match.get("similarity", sim) or 0.0))
    score = alignment_confidence
    factors: list[str] = []
    method = match.get("match_method")
    text = str(match.get("matched_text") or "")

    if method == "chunk_fallback":
        score -= 0.15
        factors.append("chunk_fallback \u22120.15")
    elif method == "sequence_correction":
        score -= 0.10
        factors.append("sequence_correction \u22120.10")

    text_len = len(text.strip())
    if text_len < 30:
        score -= 0.25
        factors.append("very_short_response \u22120.25")
    elif text_len < 100:
        score -= 0.10
        factors.append("short_response \u22120.10")

    if classification == "not_addressed" and text:
        response_signals = re.compile(
            r"\b(accepts?|agrees?|rejects?|does not agree|will |has already|"
            r"is already|intends?|plans? to|noted|welcomed?)\b",
            re.IGNORECASE,
        )
        if response_signals.search(text):
            score -= 0.10
            factors.append("possible_misclassification \u22120.10")

    if classification in {"accepted", "partial", "rejected"} and sim < 0.3:
        score -= 0.15
        factors.append("low_alignment_for_stance \u22120.15")

    return max(0.0, min(1.0, score)), factors


def generate_export_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for preset_id, preset in PRESETS.items():
        policy_pages = extract_pages(preset.policy_pdf)
        response_pages = extract_pages(preset.response_pdf)
        recommendations = extract_recommendations(
            policy_pages,
            response_pages_fallback=(
                response_pages if preset.allow_response_heading_recommendation_fallback else None
            ),
            inline_chapter_prefix=preset.inline_recommendation_chapter_prefix,
            select_committee_section=preset.select_committee_conclusions_section,
        )
        units = extract_response_units(response_pages)
        matches = match_recommendations_to_response_units(
            recommendations,
            units,
            top_k=3,
            similarity_threshold=0.05,
        )

        by_rec: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for match in matches:
            by_rec[int(match["rec_id"])].append(dict(match))

        for rec in recommendations:
            rec_id = int(rec["rec_id"])
            rec_matches = sorted(
                by_rec.get(rec_id, []),
                key=lambda item: float(item.get("similarity", 0.0)),
                reverse=True,
            )
            best = rec_matches[0] if rec_matches else None
            sim = float((best or {}).get("similarity", 0.0))
            if best:
                raw_label, class_conf = classify_with_confidence(str(best.get("matched_text") or ""))
                best_label = normalize_label(raw_label)
            else:
                best_label, class_conf = "not_addressed", 0.0
            classification = LABEL_MAP.get(best_label, "not_addressed")
            confidence, factors = _overall_confidence(best, classification, sim)
            alignment_confidence = (
                float(best.get("alignment_confidence", sim)) if best else 0.0
            )

            rows.append(
                {
                    "id": rec.get("item_label") if rec.get("item_label") is not None else rec_id,
                    "document_pair": preset_id,
                    "recommendation_page": rec.get("page_number"),
                    "recommendation_text": rec.get("text"),
                    "matched_response_page": best.get("page_number") if best else None,
                    "matched_response_text": best.get("matched_text") if best else None,
                    "classification": classification,
                    "confidence": _fixed(confidence),
                    "debug": {
                        "alignment_confidence": alignment_confidence,
                        "alignment_method": best.get("match_method") if best else None,
                        "lexical_similarity": sim if best else None,
                        "confidence_factors": factors,
                        "classification_confidence": class_conf,
                        "classifier_method": "rule_based",
                    },
                }
            )
    return rows


def _counts(rows: list[dict[str, Any]]) -> Counter:
    return Counter(str(row.get("document_pair") or "") for row in rows)


def _blank_ids(rows: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    return [
        (str(row.get("document_pair") or ""), row.get("id"))
        for row in rows
        if row.get("id") is None or str(row.get("id")).strip() == ""
    ]


def _duplicates(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    ids_by_pair: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        ids_by_pair[str(row.get("document_pair") or "")].append(str(row.get("id") or ""))
    return {
        pair: sorted([item for item, count in Counter(ids).items() if count > 1])
        for pair, ids in ids_by_pair.items()
        if any(count > 1 for count in Counter(ids).values())
    }


def metadata_failures(rows: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for row in rows:
        key = " ".join(_row_key(row))
        debug = row.get("debug")
        if not isinstance(debug, dict):
            failures.append(f"{key}: debug missing/not object")
            continue
        for field in DEBUG_FIELDS:
            if field not in debug:
                failures.append(f"{key}: debug.{field} missing")
        if "confidence_factors" in debug and not isinstance(debug["confidence_factors"], list):
            failures.append(f"{key}: debug.confidence_factors is not a list")
        if debug.get("classifier_method") != "rule_based":
            failures.append(f"{key}: debug.classifier_method is {debug.get('classifier_method')!r}")
    return failures


def response_leakage_failures(rows: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for row in rows:
        pair, item_id = _row_key(row)
        text = str(row.get("matched_response_text") or "").lstrip()
        for pattern in LEAK_START_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                failures.append(f"{pair} {item_id}: matched_response_text starts with leak pattern {pattern!r}")
        if re.search(r"\bGrowing\s+the\s+UK(?:['\u2019]s)?\s+space\s+economy\b", text, re.IGNORECASE):
            failures.append(f"{pair} {item_id}: matched_response_text contains Growing the UK's space economy")
        if (pair, item_id) in RESPONSE_TEXT_ALLOWED and pair == "blood_inquiry":
            for pattern in BLOOD_LEAK_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    failures.append(f"{pair} {item_id}: matched_response_text contains {pattern!r}")
    return failures


def recommendation_cleanup_failures(rows: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for row in rows:
        if _row_key(row) != ("space_economy", "30"):
            continue
        text = str(row.get("recommendation_text") or "")
        if re.search(r"\bGrowing\s+the\s+UK(?:['\u2019]s)?\s+space\s+economy\b", text, re.IGNORECASE):
            failures.append("space_economy 30: recommendation_text still contains trailing section heading")
    return failures


def classification_failures(rows: list[dict[str, Any]]) -> list[str]:
    by_key = {_row_key(row): row for row in rows}
    failures: list[str] = []
    for key, expected in EXPECTED_CLASSIFICATIONS.items():
        actual = by_key.get(key, {}).get("classification")
        if actual != expected:
            failures.append(f"{key[0]} {key[1]}: expected {expected}, got {actual!r}")
    return failures


def _get_field(row: dict[str, Any], field: str) -> Any:
    if field.startswith("debug."):
        debug = row.get("debug")
        if not isinstance(debug, dict):
            return MISSING
        key = field.split(".", 1)[1]
        return debug[key] if key in debug else MISSING
    return row[field] if field in row else MISSING


def _is_allowed_diff(key: tuple[str, str], field: str, before: Any, after: Any) -> bool:
    if field == "debug.classifier_method":
        return after == "rule_based"
    if field == "matched_response_text":
        return key in RESPONSE_TEXT_ALLOWED
    if field == "recommendation_text":
        return key in RECOMMENDATION_TEXT_ALLOWED
    if field == "classification":
        return key in CLASSIFICATION_ALLOWED
    if field in {"confidence", "debug.classification_confidence", "debug.confidence_factors"}:
        return key in CONFIDENCE_ALLOWED
    return False


def _format_value(value: Any, limit: int = 240) -> str:
    if value is MISSING:
        return "<missing>"
    text = ascii(value)
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def compare_to_golden(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[tuple[str, str]], list[tuple[str, str]]]:
    baseline = {_row_key(row): row for row in baseline_rows}
    candidate = {_row_key(row): row for row in candidate_rows}
    missing = sorted(set(baseline) - set(candidate))
    extra = sorted(set(candidate) - set(baseline))
    allowed: list[dict[str, Any]] = []
    unexpected: list[dict[str, Any]] = []

    for key in sorted(set(baseline) & set(candidate)):
        for field in COMPARE_FIELDS:
            before = _get_field(baseline[key], field)
            after = _get_field(candidate[key], field)
            if before == after:
                continue
            diff = {"key": key, "field": field, "before": before, "after": after}
            if _is_allowed_diff(key, field, before, after):
                allowed.append(diff)
            else:
                unexpected.append(diff)
    return allowed, unexpected, missing, extra


def _print_diff_list(title: str, diffs: list[dict[str, Any]], limit: int | None = None) -> None:
    print(title)
    shown = diffs if limit is None else diffs[:limit]
    if not shown:
        print("  none")
        return
    for diff in shown:
        pair, item_id = diff["key"]
        print(
            f"  {pair} {item_id}: {diff['field']} "
            f"{_format_value(diff['before'])} -> {_format_value(diff['after'])}"
        )
    if limit is not None and len(diffs) > limit:
        print(f"  ... {len(diffs) - limit} more")


def print_summary(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    allowed: list[dict[str, Any]],
    unexpected: list[dict[str, Any]],
    missing: list[tuple[str, str]],
    extra: list[tuple[str, str]],
) -> None:
    print(f"total row count: {len(candidate_rows)}")
    print(f"row count per document_pair: {dict(_counts(candidate_rows))}")
    print(f"duplicate ID check: {'ok' if not _duplicates(candidate_rows) else _duplicates(candidate_rows)}")
    print(f"blank ID check: {'ok' if not _blank_ids(candidate_rows) else _blank_ids(candidate_rows)}")

    leakage = response_leakage_failures(candidate_rows)
    rec_cleanup = recommendation_cleanup_failures(candidate_rows)
    metadata = metadata_failures(candidate_rows)
    classification = classification_failures(candidate_rows)

    print(f"response leakage check: {'ok' if not leakage else 'failed'}")
    for item in leakage:
        print(f"  {item}")
    for item in rec_cleanup:
        print(f"  {item}")
    print(f"metadata completeness check: {'ok' if not metadata else 'failed'}")
    for item in metadata:
        print(f"  {item}")
    print(f"classification correction check: {'ok' if not classification else 'failed'}")
    for item in classification:
        print(f"  {item}")

    print("golden diff summary:")
    print(f"  missing rows: {missing or 'none'}")
    print(f"  extra rows: {extra or 'none'}")
    print(f"  allowed field diffs: {len(allowed)}")
    print(f"  unexpected field diffs: {len(unexpected)}")

    classifier_method_added = [
        diff for diff in allowed if diff["field"] == "debug.classifier_method"
    ]
    if classifier_method_added:
        print(f"  debug.classifier_method added as rule_based on {len(classifier_method_added)} rows")

    intentional = [diff for diff in allowed if diff["field"] != "debug.classifier_method"]
    _print_diff_list("exact intentionally changed rows:", intentional)

    if unexpected:
        _print_diff_list("unexpected diffs:", unexpected)
    else:
        print("all non-listed rows are unchanged")

    failures = leakage + rec_cleanup + metadata + classification
    if failures or missing or extra or unexpected:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--generate", type=Path, help="Generate a fresh export JSON at this path before validating.")
    args = parser.parse_args()

    baseline_rows = _load_export(args.baseline)
    if args.generate:
        candidate_rows = generate_export_rows()
        _write_export(candidate_rows, args.generate)
    elif args.candidate:
        candidate_rows = _load_export(args.candidate)
    else:
        candidate_rows = generate_export_rows()

    allowed, unexpected, missing, extra = compare_to_golden(baseline_rows, candidate_rows)
    print_summary(baseline_rows, candidate_rows, allowed, unexpected, missing, extra)


if __name__ == "__main__":
    main()
