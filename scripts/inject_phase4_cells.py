"""
Phase 4 notebook injection: append Section 12 (Improved Chunking) cells.

Run from the project root:
    python scripts/inject_phase4_cells.py
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
nb_path = project_root / "notebooks" / "COP509_Notebook1_Search.ipynb"

CELLS = [
    # ---- Section header ----
    {
        "cell_type": "markdown",
        "source": """\
---
## 12. Phase 4 — Improved Chunking

Phase 1 error analysis identified **chunk_too_small** as a recurring failure
category: short page fragments (titles, section labels, captions) produce
retrieval noise and cause relevant content to be cut off mid-sentence.

Phase 4 addresses this with two improvements, both in `src/chunking.py`:

| Component | Function | Effect |
|---|---|---|
| Larger windows | `chunk_pages_v2(chunk_size=400)` | Richer per-chunk context; fewer boundary splits |
| Short-chunk merging | `_merge_short_chunks(min_words=60)` | Fragments < 60 words absorbed into neighbours |

A matching corpus builder `build_qa_corpus_v2` in `src/qa_matrix.py` uses
`chunk_pages_v2` instead of `chunk_pages` so the Phase 4 QA run uses the same
improved pipeline end-to-end.

`expand_with_context` (also new in `src/search.py`) retrieves up to one
neighbouring chunk on each side of a matched result for display — used by the
Phase 5 widget's expandable context panel.\
""",
    },

    # ---- Short-chunk analysis ----
    {
        "cell_type": "markdown",
        "source": "### Step 15 — Baseline short-chunk audit",
    },
    {
        "cell_type": "code",
        "source": """\
# Count chunks below the merge threshold in the Phase 1 baseline corpus
import pandas as pd

MIN_WORDS = 60

def _word_count(text):
    return len(text.split())

baseline_words = [_word_count(c["text"]) for c in qa_corpus]
n_short = sum(1 for w in baseline_words if w < MIN_WORDS)
pct_short = 100 * n_short / len(baseline_words)

print(f"Baseline corpus  : {len(qa_corpus):,} chunks")
print(f"Short chunks (<{MIN_WORDS} words): {n_short:,}  ({pct_short:.1f} %)")
print(f"Avg words/chunk  : {sum(baseline_words)/len(baseline_words):.1f}")

# Histogram of chunk sizes
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 3))
ax.hist(baseline_words, bins=40, color="steelblue", edgecolor="white", linewidth=0.4)
ax.axvline(MIN_WORDS, color="red", linestyle="--", label=f"merge threshold ({MIN_WORDS} words)")
ax.set_xlabel("Words per chunk")
ax.set_ylabel("Number of chunks")
ax.set_title("Baseline chunk-size distribution (200-word windows)")
ax.legend()
plt.tight_layout()
plt.show()\
""",
    },

    # ---- Build v2 corpus ----
    {
        "cell_type": "markdown",
        "source": "### Step 16 — Build Phase 4 corpus (`chunk_pages_v2`)",
    },
    {
        "cell_type": "code",
        "source": """\
from src.qa_matrix import build_qa_corpus_v2, discover_pdfs as _discover_pdfs

_all_pdfs_v2 = _discover_pdfs(DATA_DIR)
qa_corpus_v2 = build_qa_corpus_v2(
    _all_pdfs_v2,
    chunk_size=400,
    overlap=50,
    min_chunk_words=60,
)

v2_words = [len(c["text"].split()) for c in qa_corpus_v2]
n_short_v2 = sum(1 for w in v2_words if w < MIN_WORDS)

print(f"V2 corpus         : {len(qa_corpus_v2):,} chunks  (baseline: {len(qa_corpus):,})")
print(f"Short chunks (<{MIN_WORDS}): {n_short_v2:,}  (baseline: {n_short:,})")
print(f"Avg words/chunk   : {sum(v2_words)/len(v2_words):.1f}  (baseline: {sum(baseline_words)/len(baseline_words):.1f})")

fig, axes = plt.subplots(1, 2, figsize=(14, 3), sharey=False)
for ax, words, label, color in [
    (axes[0], baseline_words, "Baseline (200-word)", "steelblue"),
    (axes[1], v2_words,       "Phase 4 v2 (400-word + merge)", "seagreen"),
]:
    ax.hist(words, bins=40, color=color, edgecolor="white", linewidth=0.4)
    ax.axvline(MIN_WORDS, color="red", linestyle="--")
    ax.set_xlabel("Words per chunk")
    ax.set_ylabel("Number of chunks")
    ax.set_title(label)

plt.suptitle("Chunk-size distributions: baseline vs Phase 4 v2", y=1.02)
plt.tight_layout()
plt.show()\
""",
    },

    # ---- Run QA matrix v2 ----
    {
        "cell_type": "markdown",
        "source": """\
### Step 17 — Run QA matrix with Phase 4 corpus

This cell runs keyword and hybrid search against the Phase 4 v2 corpus.
Hybrid requires `qa_embeddings_v2`; if embeddings are not yet available the
cell falls back to keyword-only mode gracefully.\
""",
    },
    {
        "cell_type": "code",
        "source": """\
from src.qa_matrix import run_qa_matrix

# ---- Keyword mode on v2 corpus ----
_, results_v2_kw, metrics_v2_kw = run_qa_matrix(
    data_dir=DATA_DIR,
    query_bank_path=QA_QUERY_PATH,
    chunk_size=400,
    overlap=50,
    top_k=5,
    modes=["keyword"],
    prebuilt_corpus=qa_corpus_v2,
)
# Tag with phase label so the comparison table is readable
results_v2_kw["mode"] = "keyword_v2"
metrics_v2_kw["mode"] = "keyword_v2"

print(f"V2 keyword results: {len(results_v2_kw)} rows")

# ---- Hybrid mode on v2 corpus (if embeddings available) ----
try:
    from src.search import build_embeddings
    print("Building v2 corpus embeddings (may take ~30 s)…")
    qa_embeddings_v2 = build_embeddings(qa_corpus_v2)
    print(f"Embeddings: {qa_embeddings_v2.shape}")

    _, results_v2_hybrid, metrics_v2_hybrid = run_qa_matrix(
        data_dir=DATA_DIR,
        query_bank_path=QA_QUERY_PATH,
        chunk_size=400,
        overlap=50,
        top_k=5,
        modes=["hybrid"],
        prebuilt_corpus=qa_corpus_v2,
        prebuilt_embeddings=qa_embeddings_v2,
    )
    results_v2_hybrid["mode"] = "hybrid_v2"
    metrics_v2_hybrid["mode"] = "hybrid_v2"
    print(f"V2 hybrid results: {len(results_v2_hybrid)} rows")
    has_v2_hybrid = True
except Exception as exc:
    print(f"Hybrid v2 skipped: {exc}")
    results_v2_hybrid = pd.DataFrame()
    metrics_v2_hybrid = pd.DataFrame()
    has_v2_hybrid = False\
""",
    },

    # ---- Comparison table ----
    {
        "cell_type": "markdown",
        "source": "### Step 18 — Phase 4 metrics comparison",
    },
    {
        "cell_type": "code",
        "source": """\
# Combine all modes into one comparison table
compare_frames = [metrics_df, metrics_v2_kw]
if has_v2_hybrid:
    compare_frames.append(metrics_v2_hybrid)

compare_df = pd.concat(compare_frames, ignore_index=True)
overall_compare = compare_df[compare_df["query_type"] == "ALL"].copy()

# Reorder to show baseline modes first
mode_order = ["keyword", "semantic", "hybrid", "keyword_v2", "hybrid_v2"]
overall_compare["_sort"] = overall_compare["mode"].map(
    {m: i for i, m in enumerate(mode_order)}
).fillna(99)
overall_compare = overall_compare.sort_values("_sort").drop(columns="_sort")

display(
    overall_compare[[
        "mode", "n_queries",
        "recall@1", "recall@3", "recall@5",
        "precision@3", "mrr", "top1_accuracy", "ndcg@5",
    ]].style
    .background_gradient(
        subset=["recall@5", "precision@3", "mrr", "top1_accuracy", "ndcg@5"],
        cmap="RdYlGn", vmin=0, vmax=1,
    )
    .set_caption("Phase 4 — all-mode metrics comparison")
)

# Bar chart
metric_cols = ["recall@5", "precision@3", "mrr", "top1_accuracy", "ndcg@5"]
plot_data = overall_compare.set_index("mode")[metric_cols]

ax = plot_data.plot(kind="bar", figsize=(12, 4), ylim=(0, 1.1), rot=25, width=0.65)
ax.set_ylabel("Score")
ax.set_title("Phase 4 — Retrieval metrics by mode and chunking strategy")
ax.grid(axis="y", linestyle=":", alpha=0.4)
plt.tight_layout()
plt.show()\
""",
    },

    # ---- Per-type comparison ----
    {
        "cell_type": "code",
        "source": """\
# Per-query-type recall@5 for baseline keyword vs v2 keyword
type_compare = (
    compare_df[compare_df["query_type"] != "ALL"]
    .pivot(index="query_type", columns="mode", values="recall@5")
    .fillna(0)
    .sort_index()
)

cols_to_show = [c for c in ["keyword", "keyword_v2", "hybrid", "hybrid_v2"] if c in type_compare.columns]
if cols_to_show:
    display(
        type_compare[cols_to_show].style
        .background_gradient(cmap="RdYlGn", vmin=0, vmax=1, axis=None)
        .format("{:.2f}")
        .set_caption("Recall@5 by query type — baseline vs Phase 4 v2")
    )\
""",
    },

    # ---- Context expansion demo ----
    {
        "cell_type": "markdown",
        "source": """\
### Step 19 — Context expansion demo

`expand_with_context` adds neighbouring chunk text to each result without
changing scores or ranking.  This is surfaced in the Phase 5 widget as an
expandable "More context" panel beneath each result card.\
""",
    },
    {
        "cell_type": "code",
        "source": """\
from src.search import keyword_search, expand_with_context

demo_query = "Recommendation 6 pandemic preparedness"
raw_hits = keyword_search(demo_query, qa_corpus_v2, top_k=3)
expanded_hits = expand_with_context(raw_hits, qa_corpus_v2, window=1)

for i, hit in enumerate(expanded_hits, 1):
    print(f"\\n{'='*70}")
    print(f"Result {i}: {hit['source']}  page {hit['page_number']}  score={hit['score']:.4f}")
    if hit["context_before"]:
        print(f"  [before] …{hit['context_before'][-150:]!r}")
    print(f"  [match ] {hit['text'][:200]!r}…")
    if hit["context_after"]:
        print(f"  [after ] {hit['context_after'][:150]!r}…")\
""",
    },

    # ---- Phase 4 summary ----
    {
        "cell_type": "markdown",
        "source": """\
### Phase 4 Summary

| Improvement | Location | Mechanism |
|---|---|---|
| `chunk_pages_v2` | `src/chunking.py` | 400-word windows + `_merge_short_chunks` |
| `_merge_short_chunks` | `src/chunking.py` | Absorbs fragments < 60 words into neighbours |
| `detect_chunk_heading` | `src/chunking.py` | Extracts leading heading for widget display |
| `build_qa_corpus_v2` | `src/qa_matrix.py` | Corpus builder using the improved chunker |
| `expand_with_context` | `src/search.py` | Adds neighbour text to results (display only) |

**Expected gains from v2 chunking:**

- **chunk_too_small** failures eliminated — fragments absorbed before indexing.
- Higher avg words/chunk → more keyword co-occurrence per TF-IDF document → better
  recall on compound and multi-keyword queries.
- Richer semantic embeddings — longer passages encode topic context that short
  fragments lose.

Phase 5 will wire `expand_with_context` and `detect_chunk_heading` into the
premium widget UI, adding expandable context panels and heading badges to result cards.\
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
            "id": f"p4_{abs(hash(src)) % 10**8:08x}",
            "metadata": {},
            "source": sl,
        }
    return {
        "cell_type": "code",
        "id": f"p4_{abs(hash(src)) % 10**8:08x}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": sl,
    }


def main() -> None:
    with open(nb_path, encoding="utf-8") as fh:
        nb = json.load(fh)

    existing_ids = {c.get("id", "") for c in nb["cells"]}
    first_id = f"p4_{abs(hash(CELLS[0]['source'])) % 10**8:08x}"
    if first_id in existing_ids:
        print("Phase 4 cells already present — nothing to do.")
        sys.exit(0)

    new_cells = [_make_cell(c) for c in CELLS]
    nb["cells"].extend(new_cells)

    with open(nb_path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)

    print(f"Injected {len(new_cells)} Phase 4 cells into {nb_path.name}")


if __name__ == "__main__":
    main()
