"""
Classification and Evaluation widget for Notebook 2, Sections 4 and 5.

Section 4 — Response Matching & Alignment
  Compact paginated table of alignment results styled identically to the
  Section 3 extraction explorer.  Alignment-state pills only:
    MATCHED / WEAK / NO RESPONSE / EXACT LABEL MATCH / SHARED RESPONSE
  Classification outcome labels (ACCEPTED / REJECTED etc.) are NOT shown here.

Section 5 — Classification & Evaluation
  Rule-based classification of aligned response texts, optional ground-truth
  comparison, and per-class metrics.  Classification labels live only here.
"""
from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Callable

import ipywidgets as widgets
import pandas as pd
from IPython.display import clear_output, display

# ---------------------------------------------------------------------------
# Colour maps
# ---------------------------------------------------------------------------

_ALIGN_COLOURS = {
    "EXACT LABEL MATCH": "#0d6efd",
    "SHARED RESPONSE":   "#6f42c1",
    "MATCHED":           "#198754",
    "WEAK":              "#fd7e14",
    "NO RESPONSE":       "#6c757d",
}

_CLASS_COLOURS = {
    "accepted":           "#198754",
    "partially_accepted": "#fd7e14",
    "rejected":           "#dc3545",
    "not_addressed":      "#6c757d",
}

_METHOD_DISPLAY = {
    "exact_label":        "Label Match",
    "structure":          "Structure",
    "semantic":           "TF-IDF",
    "chunk_fallback":     "Chunk",
    "sequence_correction": "Sequence",
}

# ---------------------------------------------------------------------------
# HTML helpers (mirroring extraction_explorer.py)
# ---------------------------------------------------------------------------

def _short_doc(doc_id: str) -> str:
    name = str(doc_id or "")
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[_\-]+", " ", name).strip().title()
    return name or "—"


def _conf_badge(conf: float) -> str:
    if conf >= 0.75:
        bg, fg = "#d1e7dd", "#0f5132"
    elif conf >= 0.40:
        bg, fg = "#fff3cd", "#664d03"
    else:
        bg, fg = "#f8d7da", "#842029"
    return (
        f"<span style='background:{bg};color:{fg};font-size:10px;"
        f"padding:1px 5px;border-radius:3px;white-space:nowrap;'>{conf:.2f}</span>"
    )


def _pill(label: str, colours: dict) -> str:
    colour = colours.get(label, "#6c757d")
    return (
        f"<span style='background:{colour};color:#fff;font-size:10px;"
        f"padding:1px 6px;border-radius:3px;white-space:nowrap;font-weight:600;'>"
        f"{html.escape(str(label))}</span>"
    )


def _sim_bar(sim: float) -> str:
    pct = min(100, int(sim * 100))
    colour = "#28a745" if pct >= 50 else "#fd7e14" if pct >= 20 else "#dc3545"
    return (
        f"<span style='display:inline-block;width:60px;height:8px;background:#e9ecef;"
        f"border-radius:4px;margin-left:4px;vertical-align:middle;'>"
        f"<span style='display:block;width:{pct}%;height:8px;"
        f"background:{colour};border-radius:4px;'></span></span>"
    )


# ---------------------------------------------------------------------------
# Alignment status derivation
# ---------------------------------------------------------------------------

def _align_status(match: dict | None, shared_ids: set) -> str:
    """
    Derive the alignment-state pill for one recommendation row.

    Precedence (highest first):
      EXACT LABEL MATCH  match_method == exact_label
      SHARED RESPONSE    chunk shared by ≥2 recommendations
      MATCHED            method == structure OR similarity ≥ 0.35
      WEAK               some match but similarity < 0.35
      NO RESPONSE        no usable match
    """
    if match is None:
        return "NO RESPONSE"
    method = str(match.get("match_method", "") or "")
    sim = float(match.get("similarity", 0) or 0)
    cid = match.get("matched_chunk_id")
    if method == "exact_label":
        return "EXACT LABEL MATCH"
    if cid is not None and cid in shared_ids:
        return "SHARED RESPONSE"
    if method == "structure" or sim >= 0.35:
        return "MATCHED"
    if sim > 0:
        return "WEAK"
    return "NO RESPONSE"


# ---------------------------------------------------------------------------
# Build one display row per recommendation
# ---------------------------------------------------------------------------

def _build_rows(
    recommendations: list[dict],
    alignments: list[dict],
) -> list[dict]:
    by_rec: dict[int, list[dict]] = {}
    for m in alignments:
        rid = int(m.get("rec_id", -1))
        by_rec.setdefault(rid, []).append(dict(m))
    for rid in by_rec:
        by_rec[rid].sort(
            key=lambda m: (
                0 if m.get("match_method") == "exact_label" else 1,
                -float(m.get("similarity", 0) or 0),
            )
        )

    chunk_to_recs: dict[int, set] = {}
    for m in alignments:
        cid = m.get("matched_chunk_id")
        if cid is not None:
            chunk_to_recs.setdefault(int(cid), set()).add(int(m.get("rec_id", -1)))
    shared_ids = {cid for cid, rids in chunk_to_recs.items() if len(rids) > 1}

    rows: list[dict] = []
    for rec in recommendations:
        rid = int(rec.get("rec_id", -1))
        matches = by_rec.get(rid, [])
        primary = matches[0] if matches else None
        rows.append({
            "rec_id":          rid,
            "item_label":      rec.get("item_label", ""),
            "document":        rec.get("document", ""),
            "page_number":     rec.get("page_number", ""),
            "rec_text":        rec.get("text", ""),
            "match_method":    _METHOD_DISPLAY.get(
                primary.get("match_method", "") if primary else "", "—"
            ),
            "match_method_raw": (primary.get("match_method", "") if primary else ""),
            "similarity":      float(primary.get("similarity", 0) or 0) if primary else 0.0,
            "alignment_confidence": (
                float(primary.get("alignment_confidence", 0) or 0) if primary else 0.0
            ),
            "matched_text":    (primary.get("matched_text", "") if primary else ""),
            "match_source":    (primary.get("source", "") if primary else ""),
            "match_page":      (primary.get("page_number") if primary else None),
            "quoted_rec_text": (primary.get("quoted_recommendation_text", "") if primary else ""),
            "heading_text":    (primary.get("heading_text", "") if primary else ""),
            "align_status":    _align_status(primary, shared_ids),
            "_all_matches":    matches,
        })
    return rows


# ---------------------------------------------------------------------------
# Section 4 table HTML
# ---------------------------------------------------------------------------

def _s4_table_html(rows: list[dict], start_rank: int) -> str:
    head = (
        "<table style='width:100%;border-collapse:collapse;font-family:sans-serif;"
        "font-size:12px;table-layout:fixed;'>"
        "<colgroup>"
        "<col style='width:4%;'>"
        "<col style='width:8%;'>"
        "<col style='width:16%;'>"
        "<col style='width:5%;'>"
        "<col style='width:10%;'>"
        "<col style='width:7%;'>"
        "<col style='width:14%;'>"
        "<col style='width:36%;'>"
        "</colgroup>"
        "<thead><tr style='background:#f8f9fa;border-bottom:2px solid #dee2e6;'>"
        "<th style='padding:6px 8px;text-align:left;'>#</th>"
        "<th style='padding:6px 8px;text-align:left;'>Label</th>"
        "<th style='padding:6px 8px;text-align:left;'>Document</th>"
        "<th style='padding:6px 8px;text-align:left;'>Page</th>"
        "<th style='padding:6px 8px;text-align:left;'>Method</th>"
        "<th style='padding:6px 8px;text-align:left;'>Conf</th>"
        "<th style='padding:6px 8px;text-align:left;'>Match Status</th>"
        "<th style='padding:6px 8px;text-align:left;'>Response Preview</th>"
        "</tr></thead><tbody>"
    )
    parts: list[str] = []
    for i, row in enumerate(rows):
        bg = "#fff" if i % 2 == 0 else "#f8f9fa"
        rank = start_rank + i
        label = html.escape(str(row.get("item_label", "") or ""))
        doc = html.escape(_short_doc(row.get("document", "")))
        page = html.escape(str(row.get("page_number", "") or ""))
        method = html.escape(str(row.get("match_method", "") or "—"))
        conf = float(row.get("alignment_confidence", 0) or 0)
        status = str(row.get("align_status", "NO RESPONSE"))
        resp_raw = str(row.get("matched_text", ""))
        resp_prev = html.escape(resp_raw[:110]) + ("…" if len(resp_raw) > 110 else "")
        parts.append(
            f"<tr style='background:{bg};border-bottom:1px solid #e9ecef;vertical-align:top;'>"
            f"<td style='padding:5px 8px;color:#6c757d;'>{rank}</td>"
            f"<td style='padding:5px 8px;font-family:monospace;overflow:hidden;'>{label}</td>"
            f"<td style='padding:5px 8px;font-weight:500;overflow:hidden;"
            f"text-overflow:ellipsis;white-space:nowrap;'>{doc}</td>"
            f"<td style='padding:5px 8px;color:#6c757d;'>{page}</td>"
            f"<td style='padding:5px 8px;color:#495057;overflow:hidden;"
            f"text-overflow:ellipsis;white-space:nowrap;font-size:11px;'>{method}</td>"
            f"<td style='padding:5px 8px;'>{_conf_badge(conf)}</td>"
            f"<td style='padding:5px 8px;'>{_pill(status, _ALIGN_COLOURS)}</td>"
            f"<td style='padding:5px 8px;color:#495057;line-height:1.4;"
            f"overflow:hidden;font-size:11px;'>{resp_prev}</td>"
            f"</tr>"
        )
    return head + "".join(parts) + "</tbody></table>"


# ---------------------------------------------------------------------------
# Section 4 detail panel HTML
# ---------------------------------------------------------------------------

def _s4_detail_html(row: dict) -> str:
    label = str(row.get("item_label", "") or "—")
    doc = _short_doc(row.get("document", ""))
    page = row.get("page_number", "—")
    method = str(row.get("match_method", "") or "—")
    conf = float(row.get("alignment_confidence", 0) or 0)
    rec_text = str(row.get("rec_text", ""))
    status = str(row.get("align_status", "NO RESPONSE"))

    rec_block = (
        f"<div style='background:#f0f4f8;border:1px solid #dee2e6;border-radius:6px;"
        f"padding:12px 16px;margin:8px 0 4px;font-family:sans-serif;font-size:12px;'>"
        f"<div style='font-weight:700;font-size:13px;margin-bottom:4px;color:#212529;'>"
        f"Recommendation <code>{html.escape(str(label))}</code>"
        f"&nbsp;<span style='font-weight:400;color:#6c757d;'>"
        f"— {html.escape(doc)}, p.&thinsp;{html.escape(str(page))}"
        f"</span></div>"
        f"<div style='color:#495057;margin-bottom:6px;'>"
        f"{html.escape(method)}&nbsp;|&nbsp;confidence:&nbsp;{_conf_badge(conf)}"
        f"&nbsp;|&nbsp;{_pill(status, _ALIGN_COLOURS)}"
        f"</div>"
        f"<div style='background:#fff;border-radius:4px;padding:8px 10px;line-height:1.65;"
        f"color:#212529;border:1px solid #e9ecef;white-space:pre-wrap;'>"
        f"{html.escape(rec_text)}</div>"
        f"</div>"
    )

    matched_text = str(row.get("matched_text", ""))
    if not matched_text:
        resp_block = (
            "<div style='font-family:sans-serif;font-size:12px;color:#6c757d;"
            "padding:6px 0;'>No matched response found.</div>"
        )
    else:
        match_source = _short_doc(row.get("match_source", ""))
        match_page = row.get("match_page", "—")
        sim = float(row.get("similarity", 0) or 0)
        resp_block = (
            f"<div style='background:#fff8f0;border:1px solid #dee2e6;border-radius:6px;"
            f"padding:12px 16px;margin:4px 0;font-family:sans-serif;font-size:12px;'>"
            f"<div style='font-weight:600;font-size:12px;margin-bottom:4px;color:#495057;'>"
            f"Matched Response"
            f"&nbsp;<span style='font-weight:400;color:#6c757d;'>"
            f"— {html.escape(match_source)}, p.&thinsp;{html.escape(str(match_page))}"
            f"</span></div>"
            f"<div style='color:#495057;margin-bottom:6px;'>"
            f"similarity:&nbsp;<code>{sim:.4f}</code>{_sim_bar(sim)}"
            f"</div>"
            f"<div style='background:#fff;border-radius:4px;padding:8px 10px;line-height:1.65;"
            f"color:#212529;border:1px solid #e9ecef;white-space:pre-wrap;'>"
            f"{html.escape(matched_text[:600])}{'…' if len(matched_text) > 600 else ''}"
            f"</div></div>"
        )

    evidence_html = ""
    quoted = str(row.get("quoted_rec_text", "") or "")
    heading = str(row.get("heading_text", "") or "")
    if quoted or heading:
        inner = ""
        if heading:
            inner += (
                f"<div style='color:#6c757d;font-size:11px;margin-bottom:4px;'>"
                f"<b>Heading:</b> {html.escape(heading)}</div>"
            )
        if quoted:
            inner += (
                f"<div style='background:#f8f9fa;border-radius:4px;padding:6px 8px;"
                f"font-size:11px;color:#495057;border-left:3px solid #dee2e6;"
                f"white-space:pre-wrap;'>"
                f"{html.escape(quoted[:400])}{'…' if len(quoted) > 400 else ''}</div>"
            )
        evidence_html = (
            f"<details style='margin:4px 0;font-family:sans-serif;font-size:12px;'>"
            f"<summary style='cursor:pointer;color:#0d6efd;user-select:none;"
            f"padding:4px 0;'>Source / Evidence</summary>"
            f"<div style='padding:8px 12px;border:1px solid #e9ecef;border-radius:4px;"
            f"margin-top:4px;background:#fafafa;'>{inner}</div></details>"
        )

    return rec_block + resp_block + evidence_html


# ---------------------------------------------------------------------------
# Section 5 — Classification & Evaluation (static panel)
# ---------------------------------------------------------------------------

def _build_s5(
    rows: list[dict],
    classify_fn: Callable,
    compare_fn: Callable,
    df_fn: Callable,
    load_json_fn: Callable,
    gt_path,
) -> tuple[widgets.HTML, list, list, dict | None]:
    """
    Classify aligned texts, optionally compare to GT, return (widget, preds, gt, result).
    """
    predictions: list[str] = []
    for row in rows:
        text = str(row.get("matched_text", "") or "")
        predictions.append(classify_fn(text) if text else "not_addressed")

    ground_truth: list[str] = []
    result = None

    gt_path = Path(gt_path) if gt_path else None
    if gt_path and gt_path.exists():
        try:
            gt_data = load_json_fn(gt_path)
            # Format 1: flat list of label strings, aligned by position
            if isinstance(gt_data, list) and gt_data and isinstance(gt_data[0], str):
                for i in range(len(rows)):
                    ground_truth.append(gt_data[i] if i < len(gt_data) else "not_addressed")
            # Format 2: list of dicts with item_label + label keys
            elif isinstance(gt_data, list):
                gt_map: dict[str, str] = {}
                for item in gt_data:
                    lbl = str(item.get("item_label", "") or "").strip().lower()
                    gt_lbl = str(
                        item.get("label", "")
                        or item.get("classification", "")
                        or ""
                    )
                    if lbl and gt_lbl:
                        gt_map[lbl] = gt_lbl
                for row in rows:
                    lbl = str(row.get("item_label", "") or "").strip().lower()
                    ground_truth.append(gt_map.get(lbl, "not_addressed"))
            # Format 3: dict keyed by item_label
            elif isinstance(gt_data, dict):
                gt_map = {str(k).strip().lower(): str(v) for k, v in gt_data.items()}
                for row in rows:
                    lbl = str(row.get("item_label", "") or "").strip().lower()
                    ground_truth.append(gt_map.get(lbl, "not_addressed"))
            if len(ground_truth) == len(predictions) and predictions:
                result = compare_fn(predictions, ground_truth)
        except Exception:
            ground_truth = []
            result = None

    html_parts: list[str] = []

    # --- Section header ---
    html_parts.append(
        "<div style='font-family:sans-serif;margin:14px 0 6px;'>"
        "<span style='font-size:14px;font-weight:700;color:#212529;'>"
        "Section 5 — Classification &amp; Evaluation</span>"
        "<span style='font-size:12px;color:#6c757d;margin-left:8px;'>"
        "Rule-based classification of aligned response texts</span></div>"
        "<hr style='border:none;border-top:1px solid #dee2e6;margin:6px 0 10px;'/>"
    )

    # --- Classification summary table ---
    label_counts: dict[str, int] = {}
    for p in predictions:
        label_counts[p] = label_counts.get(p, 0) + 1

    summary_rows = "".join(
        f"<tr>"
        f"<td style='padding:4px 10px;'>{_pill(lbl, _CLASS_COLOURS)}</td>"
        f"<td style='padding:4px 10px;font-family:monospace;color:#212529;'>{cnt}</td>"
        f"<td style='padding:4px 10px;color:#6c757d;font-size:11px;'>"
        f"{100*cnt/max(len(predictions),1):.0f}%</td>"
        f"</tr>"
        for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1])
    )
    html_parts.append(
        "<div style='font-family:sans-serif;font-size:12px;font-weight:600;"
        "color:#495057;margin:0 0 4px;'>Predicted Label Distribution</div>"
        "<table style='border-collapse:collapse;font-family:sans-serif;"
        "font-size:12px;margin-bottom:10px;'>"
        "<thead><tr style='background:#f8f9fa;border-bottom:2px solid #dee2e6;'>"
        "<th style='padding:5px 10px;text-align:left;'>Label</th>"
        "<th style='padding:5px 10px;text-align:left;'>Count</th>"
        "<th style='padding:5px 10px;text-align:left;'>%</th>"
        f"</tr></thead><tbody>{summary_rows}</tbody></table>"
    )

    # --- Per-recommendation classification table ---
    cls_rows = ""
    for i, (row, pred) in enumerate(zip(rows, predictions)):
        bg = "#fff" if i % 2 == 0 else "#f8f9fa"
        lbl = html.escape(str(row.get("item_label", "") or ""))
        doc = html.escape(_short_doc(row.get("document", "")))
        status = str(row.get("align_status", "NO RESPONSE"))
        gt_lbl = ground_truth[i] if i < len(ground_truth) else None
        gt_cell = (
            f"<td style='padding:4px 8px;'>{_pill(gt_lbl, _CLASS_COLOURS)}</td>"
            if gt_lbl is not None else ""
        )
        match_icon = ""
        if gt_lbl is not None:
            match_icon = (
                "<td style='padding:4px 8px;color:#198754;font-weight:700;'>✓</td>"
                if pred == gt_lbl else
                "<td style='padding:4px 8px;color:#dc3545;font-weight:700;'>✗</td>"
            )
        cls_rows += (
            f"<tr style='background:{bg};border-bottom:1px solid #e9ecef;'>"
            f"<td style='padding:4px 8px;font-family:monospace;'>{lbl}</td>"
            f"<td style='padding:4px 8px;color:#495057;white-space:nowrap;"
            f"overflow:hidden;text-overflow:ellipsis;max-width:130px;'>{doc}</td>"
            f"<td style='padding:4px 8px;'>{_pill(status, _ALIGN_COLOURS)}</td>"
            f"<td style='padding:4px 8px;'>{_pill(pred, _CLASS_COLOURS)}</td>"
            f"{gt_cell}{match_icon}"
            f"</tr>"
        )
    gt_header = (
        "<th style='padding:5px 8px;text-align:left;'>GT Label</th>"
        "<th style='padding:5px 8px;text-align:left;'></th>"
        if ground_truth else ""
    )
    html_parts.append(
        "<div style='font-family:sans-serif;font-size:12px;font-weight:600;"
        "color:#495057;margin:8px 0 4px;'>Per-Recommendation Classification</div>"
        "<table style='width:100%;border-collapse:collapse;font-family:sans-serif;"
        "font-size:12px;'>"
        "<thead><tr style='background:#f8f9fa;border-bottom:2px solid #dee2e6;'>"
        "<th style='padding:5px 8px;text-align:left;'>Label</th>"
        "<th style='padding:5px 8px;text-align:left;'>Document</th>"
        "<th style='padding:5px 8px;text-align:left;'>Align Status</th>"
        "<th style='padding:5px 8px;text-align:left;'>Predicted</th>"
        f"{gt_header}"
        f"</tr></thead><tbody>{cls_rows}</tbody></table>"
    )

    # --- Evaluation metrics (if GT available) ---
    if result is not None:
        acc = result.get("accuracy", 0)
        f1 = result.get("f1_macro", 0)
        prec = result.get("precision_macro", 0)
        rec = result.get("recall_macro", 0)
        metric_cells = "".join(
            f"<td style='padding:6px 12px;text-align:center;font-size:13px;"
            f"font-weight:700;color:#212529;'>{v:.3f}</td>"
            for v in [acc, prec, rec, f1]
        )
        metric_labels = "".join(
            f"<td style='padding:2px 12px;text-align:center;font-size:11px;"
            f"color:#6c757d;'>{lbl}</td>"
            for lbl in ["Accuracy", "Precision", "Recall", "F1 Macro"]
        )
        html_parts.append(
            "<div style='font-family:sans-serif;font-size:12px;font-weight:600;"
            "color:#495057;margin:10px 0 4px;'>Evaluation Metrics vs Ground Truth</div>"
            "<table style='border-collapse:collapse;font-family:sans-serif;"
            "font-size:12px;margin-bottom:8px;background:#f8f9fa;border-radius:6px;"
            "border:1px solid #dee2e6;'>"
            f"<tbody><tr>{metric_cells}</tr><tr>{metric_labels}</tr></tbody></table>"
        )

        # Per-class metrics
        per_class = result.get("per_class", {})
        pc_rows = "".join(
            f"<tr style='background:{'#fff' if j%2==0 else '#f8f9fa'};"
            f"border-bottom:1px solid #e9ecef;'>"
            f"<td style='padding:4px 10px;'>{_pill(lbl, _CLASS_COLOURS)}</td>"
            f"<td style='padding:4px 10px;text-align:center;'>"
            f"{m.get('precision',0):.2f}</td>"
            f"<td style='padding:4px 10px;text-align:center;'>"
            f"{m.get('recall',0):.2f}</td>"
            f"<td style='padding:4px 10px;text-align:center;'>"
            f"{m.get('f1',0):.2f}</td>"
            f"<td style='padding:4px 10px;text-align:center;color:#6c757d;'>"
            f"{int(m.get('support',0))}</td>"
            f"</tr>"
            for j, (lbl, m) in enumerate(per_class.items())
        )
        html_parts.append(
            "<table style='border-collapse:collapse;font-family:sans-serif;"
            "font-size:12px;margin-bottom:10px;'>"
            "<thead><tr style='background:#f8f9fa;border-bottom:2px solid #dee2e6;'>"
            "<th style='padding:5px 10px;text-align:left;'>Label</th>"
            "<th style='padding:5px 10px;text-align:center;'>Precision</th>"
            "<th style='padding:5px 10px;text-align:center;'>Recall</th>"
            "<th style='padding:5px 10px;text-align:center;'>F1</th>"
            "<th style='padding:5px 10px;text-align:center;'>Support</th>"
            f"</tr></thead><tbody>{pc_rows}</tbody></table>"
        )
    elif gt_path and gt_path.exists():
        html_parts.append(
            "<p style='font-family:sans-serif;font-size:12px;color:#6c757d;'>"
            "Ground truth loaded but could not be aligned to predictions "
            "(check item_label format in labels.json).</p>"
        )
    else:
        html_parts.append(
            "<p style='font-family:sans-serif;font-size:12px;color:#6c757d;'>"
            "No ground-truth file found — evaluation metrics unavailable. "
            "Place <code>labels.json</code> in <code>data/ground_truth/</code> "
            "to enable comparison.</p>"
        )

    return (
        widgets.HTML(value="".join(html_parts)),
        predictions,
        ground_truth,
        result,
    )


# ---------------------------------------------------------------------------
# Main widget entry point
# ---------------------------------------------------------------------------

def show(
    recommendations: list[dict],
    alignments: list[dict],
    align_df: "pd.DataFrame",
    classify_batch: Callable,
    compare_to_ground_truth: Callable,
    results_to_dataframe: Callable,
    load_json: Callable,
    gt_path,
) -> dict:
    """
    Render the two-section widget and return evaluation artefacts.

    Returns
    -------
    dict with keys:
        label_colours  – CSS colour map for classification labels
        predictions    – list of predicted Label strings
        ground_truth   – list of GT Label strings (empty if no GT file)
        result         – EvaluationResult dict or None
    """
    if not recommendations:
        display(widgets.HTML(
            "<p style='color:#6c757d;font-family:sans-serif;'>No recommendations to display.</p>"
        ))
        return {
            "label_colours": _CLASS_COLOURS,
            "predictions": [],
            "ground_truth": [],
            "result": None,
        }

    all_rows = _build_rows(recommendations, alignments)

    all_doc_ids = sorted({r.get("document", "") for r in recommendations})
    all_methods = sorted({r.get("match_method_raw", "") for r in all_rows if r.get("match_method_raw")})

    doc_display_to_id: dict[str, str | None] = {"All loaded documents": None}
    for d in all_doc_ids:
        doc_display_to_id[_short_doc(d)] = d

    all_statuses = list(_ALIGN_COLOURS.keys())

    # Mutable state
    state: dict = {"page": 0, "selected_rec_id": None}

    # ----- Filter widgets -----
    doc_dd = widgets.Dropdown(
        options=list(doc_display_to_id.keys()),
        value="All loaded documents",
        description="Document:",
        style={"description_width": "70px"},
        layout=widgets.Layout(width="42%"),
    )
    method_dd = widgets.Dropdown(
        options=["All methods"] + [_METHOD_DISPLAY.get(m, m) for m in all_methods],
        value="All methods",
        description="Method:",
        style={"description_width": "55px"},
        layout=widgets.Layout(width="30%"),
    )
    status_dd = widgets.Dropdown(
        options=["All statuses"] + all_statuses,
        value="All statuses",
        description="Status:",
        style={"description_width": "48px"},
        layout=widgets.Layout(width="26%"),
    )
    search_box = widgets.Text(
        placeholder="Keyword filter on recommendation text…",
        description="Search:",
        style={"description_width": "50px"},
        layout=widgets.Layout(width="100%"),
    )

    # ----- Pagination widgets -----
    page_size_dd = widgets.Dropdown(
        options=[("10", 10), ("25", 25), ("50", 50), ("All", 0)],
        value=25,
        description="Per page:",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="155px"),
    )
    prev_btn = widgets.Button(description="‹ Prev", layout=widgets.Layout(width="75px"))
    next_btn = widgets.Button(description="Next ›", layout=widgets.Layout(width="75px"))
    page_info = widgets.HTML(value="")

    # ----- Output areas -----
    stats_html = widgets.HTML(value="")
    table_out = widgets.Output()
    row_select = widgets.Dropdown(
        options=[("— select a row to inspect —", None)],
        value=None,
        description="Inspect:",
        style={"description_width": "55px"},
        layout=widgets.Layout(width="80%"),
    )
    detail_out = widgets.Output()

    # ----- Build Section 5 once (static) -----
    s5_widget, predictions, ground_truth, result = _build_s5(
        all_rows, lambda text: classify_batch([text])[0],
        compare_to_ground_truth, results_to_dataframe, load_json, gt_path,
    )

    # ----- Core logic -----

    def _filtered() -> list[dict]:
        result_rows = all_rows

        raw_doc = doc_display_to_id.get(doc_dd.value)
        if raw_doc is not None:
            result_rows = [r for r in result_rows if r.get("document") == raw_doc]

        if method_dd.value != "All methods":
            result_rows = [r for r in result_rows if r.get("match_method") == method_dd.value]

        if status_dd.value != "All statuses":
            result_rows = [r for r in result_rows if r.get("align_status") == status_dd.value]

        kw = search_box.value.strip().lower()
        if kw:
            result_rows = [
                r for r in result_rows if kw in str(r.get("rec_text", "")).lower()
            ]
        return result_rows

    def _page_size() -> int | None:
        v = int(page_size_dd.value)
        return None if v == 0 else v

    def _render_detail(rec_id: int | None) -> None:
        with detail_out:
            clear_output(wait=True)
            if rec_id is None:
                return
            row = next((r for r in all_rows if r.get("rec_id") == rec_id), None)
            if row is None:
                return
            display(widgets.HTML(_s4_detail_html(row)))

    def _render(reset_page: bool = False, reset_selection: bool = False) -> None:
        if reset_page:
            state["page"] = 0
        if reset_selection:
            state["selected_rec_id"] = None

        filtered = _filtered()
        total = len(filtered)
        ps = _page_size()

        if ps is None:
            n_pages = 1
            page_rows = filtered
            start = 0
        else:
            n_pages = max(1, (total + ps - 1) // ps)
            state["page"] = max(0, min(state["page"], n_pages - 1))
            start = state["page"] * ps
            page_rows = filtered[start: start + ps]

        end = start + len(page_rows)
        range_str = f"{start + 1}–{end}" if page_rows else "0"

        stats_html.value = (
            f"<span style='font-size:12px;color:#495057;font-family:sans-serif;'>"
            f"Showing <b>{range_str}</b> of <b>{total}</b> alignment(s)"
            f"</span>"
        )
        page_info.value = (
            f"<span style='font-size:12px;color:#495057;font-family:sans-serif;"
            f"white-space:nowrap;'>Page <b>{state['page'] + 1}</b> / <b>{n_pages}</b></span>"
        )
        prev_btn.disabled = state["page"] == 0
        next_btn.disabled = state["page"] >= n_pages - 1

        with table_out:
            clear_output(wait=True)
            if page_rows:
                display(widgets.HTML(_s4_table_html(page_rows, start + 1)))
            else:
                display(widgets.HTML(
                    "<p style='color:#6c757d;font-family:sans-serif;font-size:13px;"
                    "padding:10px 0;'>No alignments match the current filters.</p>"
                ))

        page_options: list[tuple[str, int | None]] = [
            ("— select a row to inspect —", None)
        ]
        for rank, row in enumerate(page_rows, start + 1):
            rid = row.get("rec_id")
            doc_short = _short_doc(row.get("document", ""))[:20]
            lbl = str(row.get("item_label", "") or "")
            lbl_part = f" [{lbl}]" if lbl else ""
            status = row.get("align_status", "")
            page_options.append(
                (f"#{rank}  {doc_short}{lbl_part}  [{status}]", rid)
            )

        row_select.unobserve_all()
        row_select.options = page_options

        sel_id = state["selected_rec_id"]
        current_ids = {r.get("rec_id") for r in page_rows}
        if sel_id is not None and sel_id in current_ids:
            row_select.value = sel_id
        else:
            row_select.value = None
            if sel_id is not None and not any(r.get("rec_id") == sel_id for r in filtered):
                state["selected_rec_id"] = None

        row_select.observe(_on_row_select, names="value")
        _render_detail(state["selected_rec_id"])

    # ----- Observers -----

    def _on_filter_change(change=None) -> None:
        _render(reset_page=True, reset_selection=True)

    def _on_page_size_change(change=None) -> None:
        _render(reset_page=True)

    def _on_prev(btn) -> None:
        state["page"] = max(0, state["page"] - 1)
        _render()

    def _on_next(btn) -> None:
        state["page"] += 1
        _render()

    def _on_row_select(change) -> None:
        rid = change["new"]
        state["selected_rec_id"] = rid
        _render_detail(rid)

    doc_dd.observe(_on_filter_change, names="value")
    method_dd.observe(_on_filter_change, names="value")
    status_dd.observe(_on_filter_change, names="value")
    search_box.observe(_on_filter_change, names="value")
    page_size_dd.observe(_on_page_size_change, names="value")
    prev_btn.on_click(_on_prev)
    next_btn.on_click(_on_next)
    row_select.observe(_on_row_select, names="value")

    # ----- Layout -----
    s4_header = widgets.HTML(
        "<div style='font-family:sans-serif;margin:4px 0 6px;'>"
        "<span style='font-size:14px;font-weight:700;color:#212529;'>"
        "Section 4 — Response Matching &amp; Alignment</span>"
        "<span style='font-size:12px;color:#6c757d;margin-left:8px;'>"
        "Alignment status only — classification labels are in Section 5 below</span>"
        "</div>"
        "<hr style='border:none;border-top:1px solid #dee2e6;margin:6px 0 10px;'/>"
    )
    filter_row = widgets.HBox(
        [doc_dd, method_dd, status_dd],
        layout=widgets.Layout(gap="6px", margin="0 0 4px 0"),
    )
    search_row = widgets.HBox(
        [search_box],
        layout=widgets.Layout(margin="0 0 6px 0"),
    )
    pagination_row = widgets.HBox(
        [stats_html,
         widgets.HBox(
             [page_size_dd, prev_btn, page_info, next_btn],
             layout=widgets.Layout(gap="4px", align_items="center"),
         )],
        layout=widgets.Layout(
            justify_content="space-between",
            align_items="center",
            margin="0 0 4px 0",
        ),
    )
    inspector_row = widgets.HBox(
        [row_select],
        layout=widgets.Layout(margin="6px 0 4px 0"),
    )

    root = widgets.VBox([
        s4_header,
        filter_row,
        search_row,
        pagination_row,
        table_out,
        inspector_row,
        widgets.HTML(
            "<span style='font-size:11px;color:#6c757d;font-family:sans-serif;'>"
            "Select a row above to see the recommendation, matched response, "
            "and source evidence.</span>"
        ),
        detail_out,
        widgets.HTML("<div style='margin:18px 0 0;'></div>"),
        s5_widget,
    ])

    display(root)
    _render()

    return {
        "label_colours": _CLASS_COLOURS,
        "predictions":   predictions,
        "ground_truth":  ground_truth,
        "result":        result,
    }
