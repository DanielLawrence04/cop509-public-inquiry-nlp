from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache

from backend.core.presets import PRESETS
from src.alignment import match_recommendations_to_response_units
from src.classification import classify_with_confidence
from src.extraction import extract_recommendations
from src.pdf_loader import extract_pages
from src.response_units import extract_response_units


@lru_cache(maxsize=None)
def _pipeline(preset_id: str):
    preset = PRESETS[preset_id]
    policy_pages = extract_pages(preset.policy_pdf)
    response_pages = extract_pages(preset.response_pdf)
    recs = extract_recommendations(
        policy_pages,
        response_pages_fallback=(
            response_pages if preset.allow_response_heading_recommendation_fallback else None
        ),
        inline_chapter_prefix=preset.inline_recommendation_chapter_prefix,
        select_committee_section=preset.select_committee_conclusions_section,
    )
    units = extract_response_units(response_pages)
    matches = match_recommendations_to_response_units(recs, units, top_k=1)
    by_rec = defaultdict(list)
    for match in matches:
        by_rec[int(match["rec_id"])].append(match)
    best = {}
    for rec in recs:
        rec_matches = sorted(
            by_rec.get(int(rec["rec_id"]), []),
            key=lambda item: float(item.get("similarity", 0.0)),
            reverse=True,
        )
        if rec_matches:
            best[str(rec.get("item_label"))] = rec_matches[0]
    return recs, units, best


def _label_for(match) -> str:
    label, _confidence = classify_with_confidence(match.get("matched_text") or "")
    return label


def test_space_economy_uses_text_ordinal_matching_for_known_bad_rows():
    recs, _units, best = _pipeline("space_economy")
    assert len(recs) == 40
    assert Counter(match["match_method"] for match in best.values()) == {
        "structure": 30,
        "structured_ordinal": 10,
    }

    expected_terms = {
        "4": "financial year 2025/26",
        "10": "merger into DSIT",
        "12": "one government",
        "14": "public awareness",
        "16": "spring space publication",
        "17": "six capability goals",
        "18": "ARIA",
        "25": "PNT",
        "28": "assured access",
        "29": "European Launcher Challenge",
        "30": "spaceport",
        "32": "access to finance",
        "33": "Venture Capital Fellowships",
        "34": "British Business Bank",
        "36": "forward plans",
        "38": "procurement",
        "39": "contracts",
        "47": "Space Skills Advisory Panel",
        "53": "space cluster network",
        "72": "ESA",
    }
    for label, term in expected_terms.items():
        assert label in best
        assert term.lower() in best[label]["matched_text"].lower()

    assert _label_for(best["5"]) == "partially_accepted"
    assert _label_for(best["47"]) != "not_addressed"
    assert _label_for(best["72"]) != "not_addressed"
    assert _label_for(best["25"]) != "accepted"
    assert "317 uk space businesses" not in best["30"]["matched_text"].lower()
    assert "317 uk space businesses" not in best["53"]["matched_text"].lower()


def test_blood_inquiry_structured_grouping_replaces_chunk_fallback():
    recs, units, best = _pipeline("blood_inquiry")
    assert len(recs) == 58
    assert len({str(rec.get("item_label")) for rec in recs}) == 58
    assert len(units) >= 12
    assert set(match["match_method"] for match in best.values()) <= {
        "structured_grouped",
        "structured_section",
    }
    assert len(best) == 58
    assert _label_for(best["7c"]) in {"partially_accepted", "accepted"}
    for label, match in best.items():
        text = match.get("matched_text") or ""
        assert text.strip(), label
        _classification, confidence = classify_with_confidence(text)
        assert confidence > 0, label


def test_summer_and_grenfell_specific_refinements():
    _summer_recs, _summer_units, summer = _pipeline("summer_2024_disorder")
    assert _label_for(summer["1"]) == "partially_accepted"
    assert _label_for(summer["4"]) == "partially_accepted"
    assert _label_for(summer["2"]) == "partially_accepted"
    assert _label_for(summer["3"]) == "partially_accepted"
    assert _label_for(summer["7"]) == "accepted"

    _grenfell_recs, _grenfell_units, grenfell = _pipeline("grenfell_phase2")
    assert grenfell["14"]["match_method"] == "exact_label"
    assert _label_for(grenfell["14"]) == "partially_accepted"
