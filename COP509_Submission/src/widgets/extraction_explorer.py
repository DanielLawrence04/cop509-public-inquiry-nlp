"""
Extraction explorer widget for Notebook 2.

Interactive paginated table of extracted recommendations with:
  - Document, method, confidence, and keyword filters
  - Pagination (default 25 rows; options 10 / 25 / 50 / All)
  - Row click -> recommendation detail card with aligned responses
  - CSV / JSON export of the full filtered set (not just the current page)
"""
from __future__ import annotations

import html
import json
import re
from typing import Callable

import ipywidgets as widgets
import pandas as pd
from IPython.display import clear_output, display


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _short_doc(doc_id: str) -> str:
    name = str(doc_id or "")
    name = re.sub(r"\.pdf$", "", name, flags=re.IGNORECASE)
    name = re.sub(r"[_\-]+", " ", name).strip().title()
    return name or "—"


def _conf_badge(conf: float) -> str:
    if conf >= 0.90:
        bg, fg = "#d1e7dd", "#0f5132"
    elif conf >= 0.75:
        bg, fg = "#d1e7dd", "#0f5132"
    elif conf >= 0.60:
        bg, fg = "#fff3cd", "#664d03"
    else:
        bg, fg = "#f8d7da", "#842029"
    return (
        f"<span style='background:{bg};color:{fg};font-size:10px;"
        f"padding:1px 5px;border-radius:3px;white-space:nowrap;'>{conf:.2f}</span>"
    )


def _label_badge(label: str, colours: dict) -> str:
    colour = colours.get(label, "#6c757d")
    return (
        f"<span style='background:{colour};color:#fff;font-size:10px;"
        f"padding:1px 5px;border-radius:3px;white-space:nowrap;'>{html.escape(str(label))}</span>"
    )


def _sim_bar(sim: float) -> str:
    pct = min(100, int(sim * 100))
    colour = "#28a745" if pct >= 50 else "#fd7e14" if pct >= 20 else "#dc3545"
    return (
        f"<span style='display:inline-block;width:60px;height:8px;background:#e9ecef;"
        f"border-radius:4px;margin-left:4px;vertical-align:middle;'>"
        f"<span style='display:block;width:{pct}%;height:8px;background:{colour};border-radius:4px;'></span></span>"
    )


# ---------------------------------------------------------------------------
# HTML renderers
# ---------------------------------------------------------------------------

def _table_html(rows: list[dict], start_rank: int) -> str:
    head = (
        "<table style='width:100%;border-collapse:collapse;font-family:sans-serif;"
        "font-size:12px;table-layout:fixed;'>"
        "<colgroup>"
        "<col style='width:4%;'>"
        "<col style='width:18%;'>"
        "<col style='width:7%;'>"
        "<col style='width:5%;'>"
        "<col style='width:17%;'>"
        "<col style='width:6%;'>"
        "<col style='width:43%;'>"
        "</colgroup>"
        "<thead><tr style='background:#f8f9fa;border-bottom:2px solid #dee2e6;'>"
        "<th style='padding:6px 8px;text-align:left;'>#</th>"
        "<th style='padding:6px 8px;text-align:left;'>Document</th>"
        "<th style='padding:6px 8px;text-align:left;'>Label</th>"
        "<th style='padding:6px 8px;text-align:left;'>Page</th>"
        "<th style='padding:6px 8px;text-align:left;'>Method</th>"
        "<th style='padding:6px 8px;text-align:left;'>Conf</th>"
        "<th style='padding:6px 8px;text-align:left;'>Text</th>"
        "</tr></thead><tbody>"
    )
    parts: list[str] = []
    for i, row in enumerate(rows):
        bg = "#fff" if i % 2 == 0 else "#f8f9fa"
        doc = html.escape(_short_doc(row.get("document", "")))
        item_label = html.escape(str(row.get("item_label", "") or ""))
        page = html.escape(str(row.get("page_number", "") or ""))
        method = html.escape(str(row.get("extraction_method", "")))
        conf = float(row.get("confidence", 0))
        text_raw = str(row.get("text", ""))
        text_preview = html.escape(text_raw[:120]) + ("…" if len(text_raw) > 120 else "")
        rank = start_rank + i
        parts.append(
            f"<tr style='background:{bg};border-bottom:1px solid #e9ecef;vertical-align:top;'>"
            f"<td style='padding:5px 8px;color:#6c757d;'>{rank}</td>"
            f"<td style='padding:5px 8px;font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{doc}</td>"
            f"<td style='padding:5px 8px;font-family:monospace;overflow:hidden;'>{item_label}</td>"
            f"<td style='padding:5px 8px;color:#6c757d;'>{page}</td>"
            f"<td style='padding:5px 8px;color:#495057;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{method}</td>"
            f"<td style='padding:5px 8px;'>{_conf_badge(conf)}</td>"
            f"<td style='padding:5px 8px;color:#212529;line-height:1.4;overflow:hidden;'>{text_preview}</td>"
            f"</tr>"
        )
    return head + "".join(parts) + "</tbody></table>"


def _detail_html(
    rec: dict,
    aligned: list[dict],
    classify_fn: Callable,
    colours: dict,
) -> str:
    doc = _short_doc(rec.get("document", ""))
    item_label = rec.get("item_label", "") or "—"
    page = rec.get("page_number", "—")
    method = rec.get("extraction_method", "")
    conf = float(rec.get("confidence", 0))
    text = str(rec.get("text", ""))

    header = (
        f"<div style='background:#f0f4f8;border:1px solid #dee2e6;border-radius:6px;"
        f"padding:12px 16px;margin:10px 0 6px;font-family:sans-serif;font-size:12px;'>"
        f"<div style='font-weight:700;font-size:13px;margin-bottom:6px;color:#212529;'>"
        f"Recommendation <code>{html.escape(str(item_label))}</code> "
        f"&nbsp;<span style='font-weight:400;color:#6c757d;'>— {html.escape(doc)}, p.&thinsp;{html.escape(str(page))}</span>"
        f"</div>"
        f"<div style='color:#495057;margin-bottom:8px;'>"
        f"{html.escape(method)}&nbsp;|&nbsp;confidence: {_conf_badge(conf)}"
        f"</div>"
        f"<div style='background:#fff;border-radius:4px;padding:10px;line-height:1.65;"
        f"color:#212529;border:1px solid #e9ecef;white-space:pre-wrap;'>"
        f"{html.escape(text)}</div>"
        f"</div>"
    )

    if not aligned:
        return (
            header
            + "<p style='color:#6c757d;font-family:sans-serif;font-size:12px;'>"
            "No aligned responses found.</p>"
        )

    cards: list[str] = []
    for i, match in enumerate(aligned[:3], 1):
        sim = float(match.get("similarity", 0))
        resp_text = str(match.get("matched_text", ""))
        label_str = str(classify_fn(resp_text))
        preview = html.escape(resp_text[:400]) + ("…" if len(resp_text) > 400 else "")
        card = (
            f"<div style='border:1px solid #dee2e6;border-radius:5px;padding:8px 12px;"
            f"margin:5px 0;background:#fff;font-family:sans-serif;font-size:12px;'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center;"
            f"margin-bottom:5px;flex-wrap:wrap;gap:4px;'>"
            f"<span style='font-weight:600;color:#495057;'>Response match #{i}</span>"
            f"<span style='display:flex;align-items:center;gap:4px;'>"
            f"sim:&nbsp;<code>{sim:.4f}</code>{_sim_bar(sim)}&nbsp;{_label_badge(label_str, colours)}"
            f"</span>"
            f"</div>"
            f"<div style='color:#212529;line-height:1.55;'>{preview}</div>"
            f"</div>"
        )
        cards.append(card)

    section_head = (
        "<div style='font-family:sans-serif;font-size:12px;font-weight:600;"
        "color:#495057;margin:8px 0 4px;'>Aligned Responses</div>"
    )
    return header + section_head + "".join(cards)


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

def show(
    recommendations: list[dict],
    alignments: list[dict],
    classify_fn: Callable,
    label_colours: dict,
) -> None:
    """
    Render the interactive extraction explorer.

    Parameters
    ----------
    recommendations : list[Recommendation]
        Output of extract_recommendations().
    alignments : list[AlignedMatch]
        Output of match_recommendations_to_responses().
    classify_fn : Callable
        classify_response(text: str) -> label string.
    label_colours : dict
        Mapping of label string -> CSS colour for response label badges.
    """
    if not recommendations:
        display(widgets.HTML(
            "<p style='color:#6c757d;font-family:sans-serif;'>No recommendations to display.</p>"
        ))
        return

    recs: list[dict] = [dict(r) for r in recommendations]

    # Index alignments by rec_id, sorted by descending similarity
    align_by_rec: dict[int, list[dict]] = {}
    for match in (alignments or []):
        rid = int(match.get("rec_id", -1))
        align_by_rec.setdefault(rid, []).append(dict(match))
    for rid in align_by_rec:
        align_by_rec[rid].sort(key=lambda m: -float(m.get("similarity", 0)))

    # Unique document ids (preserving original doc_id strings for filtering)
    all_doc_ids = sorted({r.get("document", "") for r in recs})
    all_methods = sorted({r.get("extraction_method", "") for r in recs})

    # Map display-name -> raw doc_id (None = All)
    doc_display_to_id: dict[str, str | None] = {"All loaded documents": None}
    for d in all_doc_ids:
        doc_display_to_id[_short_doc(d)] = d

    # --- Mutable state ---
    state: dict = {"page": 0, "selected_rec_id": None}

    # --- Filter widgets ---
    doc_dd = widgets.Dropdown(
        options=list(doc_display_to_id.keys()),
        value="All loaded documents",
        description="Document:",
        style={"description_width": "70px"},
        layout=widgets.Layout(width="40%"),
    )
    method_dd = widgets.Dropdown(
        options=["All methods"] + all_methods,
        value="All methods",
        description="Method:",
        style={"description_width": "55px"},
        layout=widgets.Layout(width="32%"),
    )
    conf_dd = widgets.Dropdown(
        options=[
            ("Any confidence", 0.0),
            ("≥ 0.60  medium+", 0.60),
            ("≥ 0.75  high+", 0.75),
            ("≥ 0.90  very high", 0.90),
        ],
        value=0.0,
        description="Conf:",
        style={"description_width": "42px"},
        layout=widgets.Layout(width="26%"),
    )
    search_box = widgets.Text(
        placeholder="Keyword filter on recommendation text…",
        description="Search:",
        style={"description_width": "50px"},
        layout=widgets.Layout(width="62%"),
    )

    # --- Pagination widgets ---
    page_size_dd = widgets.Dropdown(
        options=[("10", 10), ("25", 25), ("50", 50), ("All", 0)],
        value=25,
        description="Per page:",
        style={"description_width": "60px"},
        layout=widgets.Layout(width="155px"),
    )
    prev_btn = widgets.Button(
        description="‹ Prev",
        layout=widgets.Layout(width="75px"),
    )
    next_btn = widgets.Button(
        description="Next ›",
        layout=widgets.Layout(width="75px"),
    )
    page_info = widgets.HTML(value="")

    # --- Export widgets ---
    export_csv_btn = widgets.Button(
        description="⬇ CSV",
        layout=widgets.Layout(width="72px"),
        tooltip="Download filtered recommendations as CSV",
    )
    export_json_btn = widgets.Button(
        description="⬇ JSON",
        layout=widgets.Layout(width="72px"),
        tooltip="Download filtered recommendations as JSON",
    )
    export_status = widgets.HTML(value="")

    # --- Output areas ---
    stats_html = widgets.HTML(value="")
    table_out = widgets.Output()

    # Row inspector: dropdown populated from the current page
    row_select = widgets.Dropdown(
        options=[("— select a recommendation to inspect —", None)],
        value=None,
        description="Inspect:",
        style={"description_width": "55px"},
        layout=widgets.Layout(width="80%"),
    )
    detail_out = widgets.Output()

    # --- Core logic ---

    def _filtered() -> list[dict]:
        result = recs

        raw_doc = doc_display_to_id.get(doc_dd.value)
        if raw_doc is not None:
            result = [r for r in result if r.get("document") == raw_doc]

        if method_dd.value != "All methods":
            result = [r for r in result if r.get("extraction_method") == method_dd.value]

        min_conf = float(conf_dd.value)
        if min_conf > 0.0:
            result = [r for r in result if float(r.get("confidence", 0)) >= min_conf]

        kw = search_box.value.strip().lower()
        if kw:
            result = [r for r in result if kw in str(r.get("text", "")).lower()]

        return result

    def _page_size() -> int | None:
        v = int(page_size_dd.value)
        return None if v == 0 else v

    def _render_detail(rec_id: int | None) -> None:
        with detail_out:
            clear_output(wait=True)
            if rec_id is None:
                return
            rec = next((r for r in recs if r.get("rec_id") == rec_id), None)
            if rec is None:
                return
            matched = align_by_rec.get(rec_id, [])
            display(widgets.HTML(_detail_html(rec, matched, classify_fn, label_colours)))

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
            page_recs = filtered
            start = 0
        else:
            n_pages = max(1, (total + ps - 1) // ps)
            state["page"] = max(0, min(state["page"], n_pages - 1))
            start = state["page"] * ps
            page_recs = filtered[start : start + ps]

        end = start + len(page_recs)

        # Stats strip
        range_str = f"{start + 1}–{end}" if page_recs else "0"
        stats_html.value = (
            f"<span style='font-size:12px;color:#495057;font-family:sans-serif;'>"
            f"Showing <b>{range_str}</b> of <b>{total}</b> recommendation(s)"
            f"</span>"
        )

        # Pagination info + button state
        page_info.value = (
            f"<span style='font-size:12px;color:#495057;font-family:sans-serif;"
            f"white-space:nowrap;'>Page <b>{state['page'] + 1}</b> / <b>{n_pages}</b></span>"
        )
        prev_btn.disabled = state["page"] == 0
        next_btn.disabled = state["page"] >= n_pages - 1

        # Table
        with table_out:
            clear_output(wait=True)
            if page_recs:
                display(widgets.HTML(_table_html(page_recs, start + 1)))
            else:
                display(widgets.HTML(
                    "<p style='color:#6c757d;font-family:sans-serif;font-size:13px;"
                    "padding:10px 0;'>No recommendations match the current filters.</p>"
                ))

        # Rebuild row inspector options from the current page
        page_options: list[tuple[str, int | None]] = [
            ("— select a recommendation to inspect —", None)
        ]
        for rank, row in enumerate(page_recs, start + 1):
            rid = row.get("rec_id")
            doc_short = _short_doc(row.get("document", ""))[:20]
            label = str(row.get("item_label", "") or "")
            label_part = f" [{label}]" if label else ""
            text_snip = str(row.get("text", ""))[:55]
            page_options.append((f"#{rank}  {doc_short}{label_part}  {text_snip}…", rid))

        # Detach observer temporarily to avoid spurious detail renders during rebuild
        row_select.unobserve_all()
        row_select.options = page_options

        # Restore selection if the selected rec is still on this page
        sel_id = state["selected_rec_id"]
        current_page_ids = {r.get("rec_id") for r in page_recs}
        if sel_id is not None and sel_id in current_page_ids:
            row_select.value = sel_id
        else:
            row_select.value = None
            if sel_id is not None and not any(r.get("rec_id") == sel_id for r in filtered):
                state["selected_rec_id"] = None

        row_select.observe(_on_row_select, names="value")

        # Re-render detail if selection is still valid
        _render_detail(state["selected_rec_id"])

    # --- Observers ---

    def _on_filter_change(change=None) -> None:
        export_status.value = ""
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

    def _on_export_csv(btn) -> None:
        filtered = _filtered()
        if not filtered:
            export_status.value = (
                "<span style='color:#842029;font-size:11px;'>Nothing to export.</span>"
            )
            return
        df = pd.DataFrame(filtered)
        csv_text = df.to_csv(index=False)
        export_status.value = (
            f"<details style='display:inline;'>"
            f"<summary style='cursor:pointer;font-size:11px;color:#0d6efd;'>"
            f"CSV ({len(filtered)} rows) — click to expand</summary>"
            f"<pre style='font-size:10px;max-height:200px;overflow:auto;"
            f"background:#f8f9fa;padding:6px;border-radius:4px;'>"
            f"{html.escape(csv_text[:4000])}"
            f"{'…' if len(csv_text) > 4000 else ''}</pre></details>"
        )

    def _on_export_json(btn) -> None:
        filtered = _filtered()
        if not filtered:
            export_status.value = (
                "<span style='color:#842029;font-size:11px;'>Nothing to export.</span>"
            )
            return
        json_text = json.dumps(filtered, indent=2, default=str)
        export_status.value = (
            f"<details style='display:inline;'>"
            f"<summary style='cursor:pointer;font-size:11px;color:#0d6efd;'>"
            f"JSON ({len(filtered)} items) — click to expand</summary>"
            f"<pre style='font-size:10px;max-height:200px;overflow:auto;"
            f"background:#f8f9fa;padding:6px;border-radius:4px;'>"
            f"{html.escape(json_text[:4000])}"
            f"{'…' if len(json_text) > 4000 else ''}</pre></details>"
        )

    # Wire observers
    doc_dd.observe(_on_filter_change, names="value")
    method_dd.observe(_on_filter_change, names="value")
    conf_dd.observe(_on_filter_change, names="value")
    search_box.observe(_on_filter_change, names="value")
    page_size_dd.observe(_on_page_size_change, names="value")
    prev_btn.on_click(_on_prev)
    next_btn.on_click(_on_next)
    export_csv_btn.on_click(_on_export_csv)
    export_json_btn.on_click(_on_export_json)
    row_select.observe(_on_row_select, names="value")

    # --- Layout ---
    filter_row = widgets.HBox(
        [doc_dd, method_dd, conf_dd],
        layout=widgets.Layout(gap="6px", margin="0 0 4px 0"),
    )
    search_row = widgets.HBox(
        [search_box,
         widgets.HBox(
             [export_csv_btn, export_json_btn],
             layout=widgets.Layout(gap="4px"),
         )],
        layout=widgets.Layout(justify_content="space-between", margin="0 0 6px 0"),
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
    export_out = widgets.Output()

    root = widgets.VBox([
        filter_row,
        search_row,
        pagination_row,
        table_out,
        inspector_row,
        widgets.HTML(
            "<span style='font-size:11px;color:#6c757d;font-family:sans-serif;'>"
            "Select a recommendation above to see its full text and aligned responses.</span>"
        ),
        detail_out,
        export_status,
    ])

    display(root)
    _render()
