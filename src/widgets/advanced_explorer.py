"""
Advanced search widget for Task 1 passage retrieval.

Features:
  * Mode toggle - keyword / semantic / hybrid when embeddings are available
  * Confidence badge - High / Medium / Low derived from score
  * Heading badge - extracted via detect_chunk_heading
  * "Why matched" row - matched terms, query type, adaptive alpha
  * Expandable context - neighbouring chunk text via expand_with_context
    (uses HTML5 <details> / <summary> so no JS dependency)

Usage
-----
>>> from src.widgets.advanced_explorer import show_advanced
>>> show_advanced(chunks)                          # keyword only
>>> show_advanced(chunks, embeddings=qa_embeddings)  # all three modes
"""

from __future__ import annotations

import html
import re
from typing import Optional

import ipywidgets as widgets
import numpy as np
from IPython.display import display

from ..chunking import Chunk, detect_chunk_heading
from ..search import (
    SearchResult,
    expand_with_context,
    hybrid_search,
    keyword_search,
    recommended_alpha,
    detect_query_type,
)

try:
    from ..search import semantic_search
    _HAS_SEMANTIC = True
except Exception:
    _HAS_SEMANTIC = False


# ---------------------------------------------------------------------------
# Stop words (shared with baseline widget)
# ---------------------------------------------------------------------------

_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "there", "this", "to", "was", "were", "will", "with",
    "about", "into", "after", "before",
})


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

def _highlight(text: str, query: str) -> str:
    escaped = html.escape(text)
    terms = sorted(
        {t for t in re.split(r"\W+", query.lower()) if len(t) >= 3 and t not in _STOP},
        key=len,
        reverse=True,
    )
    for term in terms:
        escaped = re.sub(
            re.escape(term),
            lambda m: f"<b style='background:#fff3cd;padding:0 1px;'>{m.group(0)}</b>",
            escaped,
            flags=re.IGNORECASE,
        )
    return escaped


def _score_bar(score: float) -> str:
    pct = min(100, int(score * 100))
    color = "#28a745" if pct >= 50 else "#fd7e14" if pct >= 20 else "#dc3545"
    return (
        f"<span style='display:inline-block;width:80px;height:9px;"
        f"background:#e9ecef;border-radius:5px;margin-left:5px;vertical-align:middle;'>"
        f"<span style='display:block;width:{pct}%;height:9px;"
        f"background:{color};border-radius:5px;'></span></span>"
    )


def _confidence_badge(score: float) -> str:
    if score >= 0.50:
        bg, fg, label = "#d1e7dd", "#0f5132", "High confidence"
    elif score >= 0.20:
        bg, fg, label = "#fff3cd", "#664d03", "Medium confidence"
    else:
        bg, fg, label = "#f8d7da", "#842029", "Low confidence"
    return (
        f"<span style='background:{bg};color:{fg};font-size:10px;"
        f"padding:1px 6px;border-radius:3px;margin-left:4px;'>{label}</span>"
    )


def _doc_type_badge(source: str) -> str:
    if "response" in source.lower():
        return "<span style='background:#cfe2ff;color:#084298;font-size:10px;padding:1px 5px;border-radius:3px;'>response</span>"
    return "<span style='background:#d1e7dd;color:#0f5132;font-size:10px;padding:1px 5px;border-radius:3px;'>recommendation</span>"


def _heading_badge(text: str) -> str:
    heading = detect_chunk_heading(text)
    if not heading:
        return ""
    short = html.escape(heading[:50])
    return (
        f"<span style='background:#e2d9f3;color:#432874;font-size:10px;"
        f"padding:1px 6px;border-radius:3px;margin-left:4px;'>{short}</span>"
    )


def _why_matched(query: str, result: dict, mode: str) -> str:
    text_lower = result["text"].lower()
    terms = [t for t in re.split(r"\W+", query.lower()) if len(t) >= 3 and t not in _STOP]
    matched = sorted({t for t in terms if t in text_lower})

    qtype = detect_query_type(query)
    parts: list[str] = []

    if matched:
        chips = "".join(
            f"<code style='background:#e9ecef;border-radius:3px;"
            f"padding:1px 5px;margin:0 2px;font-size:11px;'>{html.escape(t)}</code>"
            for t in matched
        )
        parts.append(f"Matched:&nbsp;{chips}")
    else:
        parts.append("<span style='color:#6c757d;'>No strong keyword overlap</span>")

    type_color = {"exact_phrase": "#084298", "numeric": "#664d03", "short": "#495057",
                  "exploratory": "#0f5132", "general": "#495057"}.get(qtype, "#495057")
    parts.append(
        f"Query&nbsp;type:&nbsp;<span style='color:{type_color};font-weight:600;"
        f"font-size:11px;'>{qtype}</span>"
    )

    if mode == "hybrid":
        alpha = recommended_alpha(query)
        parts.append(
            f"&alpha;=<code style='font-size:11px;'>{alpha:.2f}</code>"
            f"&nbsp;<span style='color:#6c757d;font-size:10px;'>"
            f"({'keyword-heavy' if alpha >= 0.7 else 'balanced' if alpha >= 0.45 else 'semantic-heavy'})"
            f"</span>"
        )

    return (
        "<div style='font-size:11px;color:#495057;padding:5px 10px;"
        "border-top:1px solid #f0f0f0;background:#fafafa;"
        "display:flex;gap:14px;flex-wrap:wrap;align-items:center;'>"
        + "&nbsp;&nbsp;".join(parts)
        + "</div>"
    )


def _context_panel(result: dict, query: str) -> str:
    before = result.get("context_before")
    after = result.get("context_after")
    if not before and not after:
        return ""

    parts: list[str] = []
    if before:
        parts.append(
            f"<div style='border-left:3px solid #dee2e6;padding:4px 10px;"
            f"color:#6c757d;font-size:11px;margin-bottom:4px;'>"
            f"<span style='font-size:10px;color:#adb5bd;'>BEFORE</span><br>"
            f"{_highlight(before[-200:], query)}&hellip;"
            f"</div>"
        )
    if after:
        parts.append(
            f"<div style='border-left:3px solid #dee2e6;padding:4px 10px;"
            f"color:#6c757d;font-size:11px;'>"
            f"<span style='font-size:10px;color:#adb5bd;'>AFTER</span><br>"
            f"&hellip;{_highlight(after[:200], query)}"
            f"</div>"
        )

    inner = "".join(parts)
    return (
        f"<details style='margin-top:6px;'>"
        f"<summary style='cursor:pointer;font-size:11px;color:#0d6efd;"
        f"user-select:none;padding:2px 0;'>Show neighbouring context</summary>"
        f"<div style='margin-top:6px;'>{inner}</div>"
        f"</details>"
    )


def _result_card_v2(rank: int, result: dict, query: str, mode: str) -> str:
    source = result["source"]
    page = result["page_number"]
    score = result["score"]

    doc_label = source.replace(".pdf", "")
    page_label = f"p.&nbsp;{page}" if page is not None else "page&nbsp;unknown"
    snippet = result["text"][:400]

    header = (
        f"<div style='display:flex;justify-content:space-between;align-items:center;"
        f"margin-bottom:6px;flex-wrap:wrap;gap:4px;'>"
        f"<span>"
        f"<span style='background:#343a40;color:#fff;border-radius:3px;"
        f"padding:2px 7px;font-weight:bold;margin-right:6px;'>#{rank}</span>"
        f"<span style='font-weight:600;color:#212529;'>{html.escape(doc_label)}</span>"
        f"&nbsp;<span style='color:#6c757d;font-size:11px;'>({page_label})</span>"
        f"&nbsp;{_doc_type_badge(source)}"
        f"{_heading_badge(result['text'])}"
        f"</span>"
        f"<span style='font-size:11px;color:#495057;display:flex;align-items:center;gap:4px;'>"
        f"score:&nbsp;<code>{score:.4f}</code>{_score_bar(score)}"
        f"{_confidence_badge(score)}"
        f"</span>"
        f"</div>"
    )

    snippet_html = (
        f"<div style='color:#212529;line-height:1.6;background:#f8f9fa;"
        f"padding:8px 10px;border-radius:4px;font-family:monospace;font-size:12px;'>"
        f"{_highlight(snippet, query)}&hellip;"
        f"</div>"
    )

    why = _why_matched(query, result, mode)
    context = _context_panel(result, query)

    return (
        f"<div style='margin:8px 0;padding:10px 14px;border:1px solid #dee2e6;"
        f"border-radius:6px;background:#fff;font-family:sans-serif;font-size:13px;'>"
        f"{header}{snippet_html}{why}"
        f"<div style='padding:0 0 0 4px;'>{context}</div>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

def show_advanced(
    chunks: list[Chunk],
    embeddings: Optional[np.ndarray] = None,
    top_k: int = 5,
    default_query: str = "",
) -> None:
    """
    Render the advanced search widget.

    Parameters
    ----------
    chunks : list[Chunk]
        Search corpus.
    embeddings : np.ndarray | None
        Pre-built L2-normalised embeddings from ``build_embeddings``.
        Required to enable semantic and hybrid modes.  If ``None``, those
        mode buttons are shown but disabled with a warning.
    top_k : int
        Default result count.
    default_query : str
        Optional query used to populate the first rendered result view.
    """
    sources = sorted({c["source"] for c in chunks})
    has_embeddings = embeddings is not None and len(embeddings) == len(chunks)
    available_modes = ["keyword"] + (["semantic", "hybrid"] if has_embeddings else [])

    # --- Widgets ---
    query_box = widgets.Text(
        value=default_query,
        placeholder="Enter a search query…",
        layout=widgets.Layout(width="52%"),
    )
    topk_slider = widgets.IntSlider(
        value=top_k, min=1, max=20, step=1,
        description="top-k",
        style={"description_width": "42px"},
        layout=widgets.Layout(width="26%"),
    )
    mode_toggle = widgets.ToggleButtons(
        options=available_modes,
        value="keyword",
        description="Mode:",
        style={"description_width": "42px",
               "button_width": "90px"},
    )
    doc_filter = widgets.Dropdown(
        options=["All documents"] + sources,
        value="All documents",
        description="Filter:",
        style={"description_width": "42px"},
        layout=widgets.Layout(width="60%"),
    )
    status_msg = (
        f"Corpus: <b>{len(chunks):,}</b> chunks &nbsp;|&nbsp; "
        f"<b>{len(sources)}</b> doc(s)"
    )
    if not has_embeddings:
        status_msg += (
            " &nbsp;|&nbsp; <span style='color:#856404;'>Semantic/hybrid unavailable"
            " — pass <code>embeddings=qa_embeddings</code></span>"
        )
    status = widgets.HTML(
        value=f"<span style='font-size:12px;color:#6c757d;'>{status_msg}</span>"
    )
    output = widgets.Output()

    def _run_search(query: str, corpus: list[Chunk], mode: str, k: int) -> list[dict]:
        if mode == "semantic" and has_embeddings:
            # Slice embeddings to match filtered corpus
            idx_map = {c["chunk_id"]: i for i, c in enumerate(chunks)}
            sub_idx = [idx_map[c["chunk_id"]] for c in corpus if c["chunk_id"] in idx_map]
            sub_emb = embeddings[sub_idx]
            raw = semantic_search(query, corpus, sub_emb, top_k=k)
        elif mode == "hybrid" and has_embeddings:
            idx_map = {c["chunk_id"]: i for i, c in enumerate(chunks)}
            sub_idx = [idx_map[c["chunk_id"]] for c in corpus if c["chunk_id"] in idx_map]
            sub_emb = embeddings[sub_idx]
            raw = hybrid_search(query, corpus, sub_emb, top_k=k)
        else:
            raw = keyword_search(query, corpus, top_k=k)
        return expand_with_context(raw, corpus, window=1)

    def _on_change(change=None):
        query = query_box.value.strip()
        if not query:
            with output:
                output.clear_output(wait=True)
            return

        selected = doc_filter.value
        corpus = chunks if selected == "All documents" else [
            c for c in chunks if c["source"] == selected
        ]
        mode = mode_toggle.value

        try:
            results = _run_search(query, corpus, mode, topk_slider.value)
        except Exception as exc:
            with output:
                output.clear_output(wait=True)
                display(widgets.HTML(
                    f"<p style='color:#842029;font-family:sans-serif;'>"
                    f"Search error ({mode}): {html.escape(str(exc))}</p>"
                ))
            return

        with output:
            output.clear_output(wait=True)
            if not results:
                display(widgets.HTML(
                    "<p style='color:#6c757d;font-family:sans-serif;'>"
                    "No results found.</p>"
                ))
                return

            doc_counts: dict[str, int] = {}
            for r in results:
                doc_counts[r["source"]] = doc_counts.get(r["source"], 0) + 1
            summary_parts = [
                f"{v}&times;&nbsp;{k_.replace('.pdf','')}"
                for k_, v in sorted(doc_counts.items(), key=lambda x: -x[1])
            ]
            qtype = detect_query_type(query)
            header_html = (
                f"<p style='margin:4px 0 8px;font-size:12px;color:#495057;"
                f"font-family:sans-serif;'>"
                f"<b>{len(results)}</b> result(s) &mdash; "
                f"mode:&nbsp;<b>{mode}</b> &nbsp;|&nbsp; "
                f"query type:&nbsp;<b>{qtype}</b>"
                + (f" &nbsp;|&nbsp; &alpha;=<b>{recommended_alpha(query):.2f}</b>" if mode == "hybrid" else "")
                + "&nbsp;&mdash;&nbsp;" + " &middot; ".join(summary_parts)
                + "</p>"
            )
            cards = "".join(
                _result_card_v2(i, r, query, mode)
                for i, r in enumerate(results, 1)
            )
            display(widgets.HTML(header_html + cards))

    query_box.observe(_on_change, names="value")
    topk_slider.observe(_on_change, names="value")
    doc_filter.observe(_on_change, names="value")
    mode_toggle.observe(_on_change, names="value")

    _on_change()
    display(widgets.VBox([
        status,
        widgets.HBox([query_box, topk_slider]),
        widgets.HBox([mode_toggle]),
        doc_filter,
        output,
    ]))
