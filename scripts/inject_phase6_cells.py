"""
Phase 6 notebook injection: append Section 14 (Final Evaluation) cells.

Run from the project root:
    python scripts/inject_phase6_cells.py
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
nb_path = project_root / "notebooks" / "COP509_Notebook1_Search.ipynb"

CELLS = [
    {
        "cell_type": "markdown",
        "source": """\
---
## 14. Phase 6 — Final Evaluation: Before/After Comparison

Phase 6 re-runs the complete QA matrix across every configuration introduced
in Phases 1–5 and assembles a single comparison table that serves as the
primary coursework evidence of system improvement.

| Baseline | Improvements applied |
|---|---|
| `keyword` — TF-IDF, 200-word chunks | Phase 2: deduplication, global corpus |
| `semantic` — MiniLM, 200-word chunks | Phase 3: hybrid ranking, adaptive α |
| `hybrid` — adaptive α, 200-word chunks | Phase 4: 400-word chunks, short-chunk merge |
| `keyword_v2` — TF-IDF, 400-word chunks | Phase 5: advanced widget (display) |
| `hybrid_v2` — adaptive α, 400-word chunks | — |

All five modes are run against the **same 50-query bank** so results are
directly comparable.  Metrics: Recall@1/3/5, Precision@3, MRR, Top-1 accuracy,
Doc accuracy, nDCG@5.\
""",
    },

    {
        "cell_type": "markdown",
        "source": "### Step 23 — Assemble full results across all modes",
    },
    {
        "cell_type": "code",
        "source": """\
# Collect all metrics DataFrames built in earlier sections.
# Each was tagged with its mode label so concat is unambiguous.
all_metrics_frames = []
all_results_frames = []

# Phase 1–3: keyword / semantic / hybrid (baseline corpus)
for _df in [metrics_df]:
    all_metrics_frames.append(_df)

# Phase 4: keyword_v2 / hybrid_v2 (v2 corpus)
for _df in [metrics_v2_kw]:
    all_metrics_frames.append(_df)

if has_v2_hybrid:
    all_metrics_frames.append(metrics_v2_hybrid)

final_metrics = pd.concat(all_metrics_frames, ignore_index=True)

# ---- Overall comparison ----
mode_order = ["keyword", "semantic", "hybrid", "keyword_v2", "hybrid_v2"]
overall_final = final_metrics[final_metrics["query_type"] == "ALL"].copy()
overall_final["_sort"] = overall_final["mode"].map(
    {m: i for i, m in enumerate(mode_order)}
).fillna(99)
overall_final = overall_final.sort_values("_sort").drop(columns="_sort")

print(f"Modes in final comparison: {overall_final['mode'].tolist()}")
display(
    overall_final[[
        "mode", "n_queries",
        "recall@1", "recall@3", "recall@5",
        "precision@3", "mrr", "top1_accuracy", "doc_accuracy", "ndcg@5",
    ]].style
    .background_gradient(
        subset=["recall@5", "precision@3", "mrr", "top1_accuracy", "ndcg@5"],
        cmap="RdYlGn", vmin=0, vmax=1,
    )
    .set_caption("Phase 6 — Final overall metrics (all modes)")
)\
""",
    },

    {
        "cell_type": "markdown",
        "source": "### Step 24 — Improvement delta table",
    },
    {
        "cell_type": "code",
        "source": """\
# Compute delta vs keyword baseline for each metric
metric_cols = ["recall@1", "recall@3", "recall@5", "precision@3",
               "mrr", "top1_accuracy", "doc_accuracy", "ndcg@5"]

baseline_row = overall_final[overall_final["mode"] == "keyword"]
if baseline_row.empty:
    print("Keyword baseline not found — run Section 10 first.")
else:
    baseline = baseline_row[metric_cols].iloc[0]

    delta_rows = []
    for _, row in overall_final.iterrows():
        mode = row["mode"]
        if mode == "keyword":
            continue
        deltas = {col: round(row[col] - baseline[col], 3) for col in metric_cols}
        deltas["mode"] = mode
        delta_rows.append(deltas)

    delta_df = pd.DataFrame(delta_rows).set_index("mode")[metric_cols]

    def _color_delta(v):
        if v > 0.005:
            return "color:#0f5132;font-weight:600"
        elif v < -0.005:
            return "color:#842029"
        return "color:#6c757d"

    display(
        delta_df.style
        .applymap(_color_delta)
        .format("{:+.3f}")
        .set_caption("Δ vs keyword baseline (green = improvement)")
    )\
""",
    },

    {
        "cell_type": "markdown",
        "source": "### Step 25 — Final bar chart",
    },
    {
        "cell_type": "code",
        "source": """\
import matplotlib.pyplot as plt

plot_cols = ["recall@5", "mrr", "top1_accuracy", "ndcg@5"]
plot_data = overall_final.set_index("mode")[plot_cols]

# Keep only modes that exist
plot_data = plot_data[[c for c in plot_cols if c in plot_data.columns]]

colors = {
    "keyword":    "#6c757d",
    "semantic":   "#0d6efd",
    "hybrid":     "#198754",
    "keyword_v2": "#fd7e14",
    "hybrid_v2":  "#6f42c1",
}
bar_colors = [colors.get(m, "#343a40") for m in plot_data.index]

ax = plot_data.plot(
    kind="bar", figsize=(13, 5), ylim=(0, 1.1), rot=0, width=0.7,
    color=["#6c757d", "#0d6efd", "#198754", "#6f42c1"],
)
ax.set_ylabel("Score")
ax.set_title("Phase 6 — Final retrieval metrics by mode and chunking strategy")
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.legend(loc="upper right", framealpha=0.9)
plt.tight_layout()
plt.show()\
""",
    },

    {
        "cell_type": "markdown",
        "source": "### Step 26 — Per-query-type final breakdown",
    },
    {
        "cell_type": "code",
        "source": """\
# Recall@5 heatmap: query type × mode
type_final = (
    final_metrics[final_metrics["query_type"] != "ALL"]
    .pivot(index="query_type", columns="mode", values="recall@5")
    .fillna(0)
    .sort_index()
)

cols_ordered = [m for m in mode_order if m in type_final.columns]
if cols_ordered:
    display(
        type_final[cols_ordered].style
        .background_gradient(cmap="RdYlGn", vmin=0, vmax=1, axis=None)
        .format("{:.2f}")
        .set_caption("Recall@5 by query type — all phases")
    )

    ax2 = type_final[cols_ordered].plot(
        kind="bar", figsize=(14, 5), ylim=(0, 1.1), rot=40, width=0.75
    )
    ax2.set_ylabel("Recall@5")
    ax2.set_title("Final Recall@5 by Query Type across All Modes")
    ax2.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()\
""",
    },

    {
        "cell_type": "markdown",
        "source": "### Step 27 — Error category analysis: before vs after",
    },
    {
        "cell_type": "code",
        "source": """\
from src.qa_matrix_report import error_analysis_table

# Collect results_df entries — use whichever modes are available
all_results_frames = [results_df]
for _rf in [results_v2_kw]:
    all_results_frames.append(_rf)
if has_v2_hybrid:
    all_results_frames.append(results_v2_hybrid)

combined_results = pd.concat(all_results_frames, ignore_index=True)

for _mode in mode_order:
    _sub = combined_results[combined_results["mode"] == _mode]
    if _sub.empty:
        continue
    _err = error_analysis_table(_sub)
    if not _err.empty:
        display(
            _err[["error_category", "count", "description", "example_query"]]
            .style.set_caption(f"Error analysis — {_mode}")
            .hide(axis="index")
        )\
""",
    },

    {
        "cell_type": "markdown",
        "source": "### Step 28 — Final findings report",
    },
    {
        "cell_type": "code",
        "source": """\
from src.qa_matrix_report import generate_findings_report, error_analysis_table
from IPython.display import Markdown

# Use best available mode for the report
best_mode = "hybrid_v2" if has_v2_hybrid else "hybrid" if "hybrid" in combined_results["mode"].values else "keyword"
best_results = combined_results[combined_results["mode"] == best_mode]
best_metrics = final_metrics[final_metrics["mode"] == best_mode]
best_errors  = error_analysis_table(best_results)

report_md = generate_findings_report(best_results, best_metrics, best_errors)
display(Markdown(report_md))\
""",
    },

    {
        "cell_type": "markdown",
        "source": """\
### Phase 6 Summary — Cumulative Improvement Evidence

The table below captures the key outcome of the phased improvement programme.
All numbers are from the 50-query evaluation bank against the same 10 documents.

| Phase | Key change | Expected recall@5 impact |
|---|---|---|
| Phase 1 | Evaluation harness + error categorisation | Baseline established |
| Phase 2 | Deduplication + global corpus | Removes duplicate result bias |
| Phase 3 | Hybrid ranking + adaptive α + boosts | +paraphrase & topic recall |
| Phase 4 | 400-word chunks + short-chunk merging | +boundary & fragment failures |
| Phase 5 | Advanced widget (display only) | No metric change; UX improvement |

**Metric targets (distinction level):**
- Recall@5 ≥ 0.80 across all query types
- MRR ≥ 0.65 (at least one relevant result usually in top 2)
- nDCG@5 ≥ 0.70 (relevant results ranked higher than irrelevant)

Where targets are not met, the error table in Step 27 identifies the remaining
failure modes and suggests further improvement directions (e.g. BM25+, re-ranking,
query expansion for short/ambiguous queries).\
""",
    },
]


def _source_lines(text: str) -> list[str]:
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def _make_cell(cell_def: dict) -> dict:
    src = cell_def["source"]
    sl = _source_lines(src)
    if cell_def["cell_type"] == "markdown":
        return {
            "cell_type": "markdown",
            "id": f"p6_{abs(hash(src)) % 10**8:08x}",
            "metadata": {},
            "source": sl,
        }
    return {
        "cell_type": "code",
        "id": f"p6_{abs(hash(src)) % 10**8:08x}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": sl,
    }


def main() -> None:
    with open(nb_path, encoding="utf-8") as fh:
        nb = json.load(fh)

    existing_ids = {c.get("id", "") for c in nb["cells"]}
    first_id = f"p6_{abs(hash(CELLS[0]['source'])) % 10**8:08x}"
    if first_id in existing_ids:
        print("Phase 6 cells already present — nothing to do.")
        sys.exit(0)

    new_cells = [_make_cell(c) for c in CELLS]
    nb["cells"].extend(new_cells)

    with open(nb_path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)

    print(f"Injected {len(new_cells)} Phase 6 cells into {nb_path.name}")


if __name__ == "__main__":
    main()
