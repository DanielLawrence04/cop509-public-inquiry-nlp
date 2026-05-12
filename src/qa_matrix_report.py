"""
QA Matrix report generator.

Consumes the DataFrames produced by ``qa_matrix.run_qa_matrix`` and
produces two further artefacts:

  error_analysis_table(results_df)
      A grouped failure table counting top-1 errors by category.

  generate_findings_report(results_df, metrics_df, error_df)
      A markdown-formatted findings report summarising what worked,
      what failed, root causes, and a prioritised fix list with evidence.
"""

from __future__ import annotations

import textwrap

import pandas as pd

from .qa_matrix import _ERROR_DESCRIPTIONS


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def error_analysis_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate top-1 failures by error category.

    Returns a DataFrame with columns:
      error_category, count, pct, description, example_query
    sorted by descending count.  Rows where auto_relevance is
    'relevant' or 'likely_relevant' are excluded — they are successes.
    """
    top1 = results_df[results_df["rank"] == 1]
    failures = top1[
        ~top1["auto_relevance"].isin(["relevant", "likely_relevant"])
        & (top1["error_category"] != "")
    ].copy()

    if failures.empty:
        return pd.DataFrame(
            columns=["error_category", "count", "pct", "description", "example_query"]
        )

    total = len(failures)
    rows = []
    for cat, group in failures.groupby("error_category"):
        example = group.iloc[0]["query"][:80]
        rows.append({
            "error_category": cat,
            "count": len(group),
            "pct": round(100 * len(group) / total, 1),
            "description": _ERROR_DESCRIPTIONS.get(str(cat), ""),
            "example_query": example,
        })

    return (
        pd.DataFrame(rows)
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def query_type_error_breakdown(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-tabulate query_type vs error_category for top-1 failures.

    Returns a pivot table (query_type × error_category) with counts.
    """
    top1 = results_df[results_df["rank"] == 1]
    failures = top1[
        ~top1["auto_relevance"].isin(["relevant", "likely_relevant"])
    ].copy()

    if failures.empty:
        return pd.DataFrame()

    return (
        failures.groupby(["query_type", "error_category"])
        .size()
        .unstack(fill_value=0)
    )


# ---------------------------------------------------------------------------
# Findings report
# ---------------------------------------------------------------------------

def generate_findings_report(
    results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    error_df: pd.DataFrame,
) -> str:
    """
    Return a markdown-formatted findings report.

    Sections
    --------
    1. Metrics summary (overall keyword + semantic if available)
    2. What worked well (best query types by recall@5)
    3. What failed (worst query types)
    4. Most common failure types with evidence
    5. Prioritised recommended fixes
    6. Improvement roadmap
    """
    lines: list[str] = []

    def _h2(title: str) -> None:
        lines.extend(["", f"## {title}", ""])

    def _h3(title: str) -> None:
        lines.extend(["", f"### {title}", ""])

    def _para(text: str) -> None:
        lines.append(text.strip())
        lines.append("")

    def _li(text: str) -> None:
        lines.append(f"- {text.strip()}")

    lines.append("# Search QA Matrix — Findings Report")
    lines.append("")
    lines.append(
        "_Auto-generated after retrieval evaluation. "
        "Relevance labels are derived from document provenance, keyword "
        "coverage (≥50% = likely_relevant) and anchor substring matching. "
        "Human grading of the `manual_relevance` column will refine these estimates._"
    )

    # --- 1. Metrics summary ---
    _h2("1. Overall Retrieval Metrics")

    for mode in ["keyword", "semantic"]:
        overall = metrics_df[
            (metrics_df["mode"] == mode) & (metrics_df["query_type"] == "ALL")
        ]
        if overall.empty:
            continue
        r = overall.iloc[0]
        n = int(r["n_queries"])
        _para(
            f"**{mode.capitalize()} search** ({n} queries): "
            f"Recall@1={r['recall@1']:.2f}, Recall@5={r['recall@5']:.2f}, "
            f"Precision@3={r['precision@3']:.2f}, MRR={r['mrr']:.2f}, "
            f"Top-1 Acc={r['top1_accuracy']:.2f}, Doc Acc={r['doc_accuracy']:.2f}, "
            f"nDCG@5={r['ndcg@5']:.2f}."
        )

    # Per-type table as a fenced block
    kw_types = metrics_df[
        (metrics_df["mode"] == "keyword") & (metrics_df["query_type"] != "ALL")
    ][["query_type", "n_queries", "recall@5", "precision@3", "mrr", "top1_accuracy"]].copy()

    if not kw_types.empty:
        _h3("Keyword Metrics by Query Type")
        lines.append(kw_types.to_string(index=False))
        lines.append("")

    # --- 2. What worked well ---
    _h2("2. What Worked Well")

    if not kw_types.empty:
        best = kw_types.sort_values("recall@5", ascending=False).head(3)
        for _, row in best.iterrows():
            _li(
                f"**{row['query_type']}** — Recall@5={row['recall@5']:.2f}, "
                f"MRR={row['mrr']:.2f}. "
                "Distinctive vocabulary or confirmed anchors aligned well with TF-IDF scoring."
            )
        lines.append("")

    # High-precision modes
    if not metrics_df.empty:
        best_p3 = metrics_df[metrics_df["query_type"] != "ALL"].sort_values(
            "precision@3", ascending=False
        ).head(1)
        if not best_p3.empty:
            r = best_p3.iloc[0]
            _li(
                f"Highest Precision@3: **{r['query_type']}** (mode={r['mode']}, "
                f"P@3={r['precision@3']:.2f}) — top results are consistently relevant."
            )
            lines.append("")

    # --- 3. What failed ---
    _h2("3. What Failed")

    if not kw_types.empty:
        worst = kw_types.sort_values("recall@5", ascending=True).head(3)
        for _, row in worst.iterrows():
            _li(
                f"**{row['query_type']}** — Recall@5={row['recall@5']:.2f}, "
                f"MRR={row['mrr']:.2f}. "
                "Retriever struggles here — likely vocabulary mismatch or thin document coverage."
            )
        lines.append("")

    # --- 4. Most common failure types ---
    _h2("4. Most Common Failure Types")

    if error_df.empty:
        _para("No top-1 failures detected. All queries returned at least a likely-relevant result.")
    else:
        for _, row in error_df.head(5).iterrows():
            _li(
                f"**{row['error_category']}** "
                f"({row['count']} failures, {row['pct']}% of failures): "
                f"{row['description']}"
            )
            lines.append(f"  - *Example:* \"{row['example_query']}\"")
        lines.append("")

    # --- 5. Recommended fixes ---
    _h2("5. Recommended Fixes (Priority Order)")
    fixes = _recommended_fixes(error_df, metrics_df)
    for i, fix in enumerate(fixes, start=1):
        lines.append(f"{i}. {fix}")
        lines.append("")

    # --- 6. Roadmap ---
    _h2("6. Improvement Roadmap")
    roadmap = textwrap.dedent("""
    | Area | Focus | Key action |
    |-------|-------|------------|
    | 2 | Retrieval correctness | Include response PDFs in global search; stable chunk ids; dedup overlapping chunks |
    | 3 | Ranking quality | Hybrid TF-IDF + semantic score; query-type-aware mode selection; doc-pair boost |
    | 4 | Chunking | Heading-preserving chunks; neighbour-context expansion; short-chunk merge |
    | 5 | Premium UI | Mode toggle, term highlighting, confidence badges, "why this matched" panel |
    | 6 | Evidence | Re-run QA matrix post-improvements; publish before/after metric and error comparison |
    """).strip()
    lines.append(roadmap)
    lines.append("")

    return "\n".join(lines)


def _recommended_fixes(
    error_df: pd.DataFrame, metrics_df: pd.DataFrame
) -> list[str]:
    """Generate a prioritised list of fix recommendations from observed errors."""
    fixes: list[str] = []

    if error_df.empty:
        # No failures: focus on incremental improvements
        overall = metrics_df[metrics_df["query_type"] == "ALL"]
        if not overall.empty:
            best_mrr = overall["mrr"].max()
            if best_mrr < 0.7:
                fixes.append(
                    "**Add hybrid ranking** (TF-IDF + semantic cosine). "
                    "MRR is below 0.7, suggesting relevant passages exist in the corpus "
                    "but are not always ranked first. A combined score would improve ordering."
                )
        fixes.append(
            "**Activate semantic search mode** for paraphrase and broad-exploratory query types. "
            "Even with low error counts, semantic retrieval recovers passages that share meaning "
            "but not exact vocabulary."
        )
        return fixes

    cats = set(error_df["error_category"].tolist())

    if "paraphrase_miss" in cats:
        count = int(error_df.loc[error_df["error_category"] == "paraphrase_miss", "count"].sum())
        fixes.append(
            f"**Enable semantic search** (MiniLM-L6-v2, already in `src/search.py`). "
            f"{count} paraphrase_miss failures occur because TF-IDF requires exact vocabulary overlap. "
            "Semantic retrieval encodes meaning rather than tokens — this is the highest-value fix. "
            "*Evidence: paraphrase_miss is a leading failure type.*"
        )

    if "ranking_issue" in cats:
        count = int(error_df.loc[error_df["error_category"] == "ranking_issue", "count"].sum())
        fixes.append(
            f"**Implement hybrid scoring** (keyword + semantic cosine). "
            f"{count} ranking_issue failures indicate that a relevant passage exists in the corpus "
            "but is beaten to rank-1 by a higher-TF-IDF chunk. A weighted hybrid "
            "boosts the correct passage without losing lexical precision. "
            "*Evidence: relevant chunk is retrievable but outranked.*"
        )

    if "wrong_doc_similar_topic" in cats:
        count = int(error_df.loc[error_df["error_category"] == "wrong_doc_similar_topic", "count"].sum())
        fixes.append(
            f"**Add document-pair awareness to ranking**. "
            f"{count} wrong_doc_similar_topic failures are caused by vocabulary overlap between "
            "inquiry reports (e.g. 'transparency', 'compensation', 'recommendation' appear across "
            "all five pairs). Boosting results that match the user's selected document pair "
            "would eliminate most of these. *Evidence: high score but wrong source document.*"
        )

    if "chunk_too_small" in cats:
        count = int(error_df.loc[error_df["error_category"] == "chunk_too_small", "count"].sum())
        fixes.append(
            f"**Filter or merge very short chunks** (< {60} words) at corpus build time. "
            f"{count} top-1 results were short fragments — likely section headings, page numbers, "
            "or OCR artefacts. Removing them would promote substantive passages. "
            "*Evidence: chunk_too_small top-1 results contain no retrievable passage text.*"
        )

    if "keyword_coverage_low" in cats:
        count = int(error_df.loc[error_df["error_category"] == "keyword_coverage_low", "count"].sum())
        fixes.append(
            f"**Expand expected_keywords in the query bank** and consider testing 300-word chunks. "
            f"{count} correct-document results show low keyword coverage, suggesting the relevant "
            "concept is split across chunk boundaries. Larger chunks would preserve more context "
            "per window. *Evidence: doc_match=True but keyword_coverage < 0.25.*"
        )

    if not fixes:
        fixes.append(
            "**Activate semantic mode** as a complementary retrieval path. "
            "Keyword search is performing well; semantic retrieval would extend coverage "
            "to paraphrase and exploratory queries without degrading precision."
        )

    return fixes
