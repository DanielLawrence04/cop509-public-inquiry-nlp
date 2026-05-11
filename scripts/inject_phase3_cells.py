"""
Phase 3 notebook injection: append Section 11 (Hybrid Ranking) cells.

Run from the project root:
    python scripts/inject_phase3_cells.py
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
## 11. Phase 3 — Hybrid Ranking Strategy

Pure TF-IDF retrieval works well when the query vocabulary matches the document
closely, but it fails for **paraphrase queries** (where the user chooses different
words from the author) and **exploratory queries** (where the user describes a
topic rather than quoting from it).  Pure semantic search handles paraphrases
well but can be over-broad on exact-phrase and numeric queries.

Phase 3 introduces three targeted ranking improvements, all in `src/search.py`:

| Component | Function | Purpose |
|---|---|---|
| Query-type detection | `detect_query_type(query)` | Classify each query into one of five types |
| Adaptive alpha | `recommended_alpha(query)` | Select keyword/semantic blend per query type |
| Hybrid scoring | `hybrid_search(query, chunks, embeddings)` | Weighted TF-IDF + semantic cosine |
| Heading boost | `_rec_heading_boost_scores` | +0.08 bonus for numbered-recommendation chunks |
| Document-pair boost | `doc_pair_boost(results, preferred_sources)` | +0.12 bonus when user has a pair in focus |

### Scoring formula

```
combined = alpha × tfidf_norm + (1 − alpha) × semantic_cosine
         + 0.15 × [exact phrase present]
         + 0.08 × [recommendation heading matches query number]
         + 0.12 × [source is in preferred document pair]
```

Where `alpha` is set per query type:

| Type | alpha | Rationale |
|---|---|---|
| exact_phrase | 0.80 | Exact tokens must fire — keyword dominant |
| numeric | 0.85 | Literal numbers handled better by TF-IDF |
| short | 0.40 | Ambiguous; semantic broadens recall |
| exploratory | 0.25 | Long queries encode meaning better as dense vectors |
| general | 0.55 | Slight keyword preference for policy vocabulary |\
""",
    },

    # ---- detect_query_type demo ----
    {
        "cell_type": "markdown",
        "source": "### Step 10 — Query-type detection",
    },
    {
        "cell_type": "code",
        "source": """\
from src.search import detect_query_type, recommended_alpha

demo_queries = [
    ("full and fair financial redress",                         "exact_phrase"),
    ("50,000 trackable pieces of orbital debris",              "numeric"),
    ("Recommendation 6 pandemic exercise",                     "exact_phrase"),
    ("transparency",                                            "short"),
    ("compensation scheme",                                     "short"),
    ("how should government effectively change public behaviour including evidence evaluation",
     "exploratory"),
    ("orbital debris risk to spacecraft",                       "general"),
    ("wrongful prosecution compensation post office",          "general"),
]

rows = []
for query, expected in demo_queries:
    qtype  = detect_query_type(query)
    alpha  = recommended_alpha(query)
    match  = "✓" if qtype == expected else f"✗ (expected {expected})"
    rows.append({
        "query"   : query[:65],
        "detected": qtype,
        "alpha"   : alpha,
        "expected": expected,
        "match"   : match,
    })

detect_df = pd.DataFrame(rows)
display(detect_df.style.set_caption("Query-type detection examples"))\
""",
    },

    # ---- Hybrid search run ----
    {
        "cell_type": "markdown",
        "source": """\
### Step 11 — Run QA matrix with hybrid mode

This cell runs the QA matrix in **hybrid mode** using the pre-built corpus and
embeddings from Step 7 (semantic mode).  If embeddings are not available the
cell catches the exception and exits gracefully.

The hybrid results are appended to `results_df` and `metrics_df` so that all
three modes (keyword, semantic, hybrid) can be compared side-by-side.\
""",
    },
    {
        "cell_type": "code",
        "source": """\
try:
    # Re-use qa_embeddings built in Step 7; skip if not yet computed
    _ = qa_embeddings
    print(f"Embeddings available: {qa_embeddings.shape} — running hybrid mode")

    _, results_hybrid, metrics_hybrid = run_qa_matrix(
        data_dir=DATA_DIR,
        query_bank_path=QA_QUERY_PATH,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        top_k=5,
        modes=["hybrid"],
        prebuilt_corpus=qa_corpus,
        prebuilt_embeddings=qa_embeddings,
    )

    results_df = pd.concat([results_df, results_hybrid], ignore_index=True)
    metrics_df = pd.concat([metrics_df, metrics_hybrid], ignore_index=True)

    print(f"Hybrid results added: {len(results_hybrid)} rows")
    print(f"Modes now in results_df: {results_df['mode'].unique().tolist()}")

except NameError:
    print("qa_embeddings not found — run Step 7 (semantic mode) first, then re-run this cell.")
except Exception as exc:
    print(f"Hybrid mode error: {exc}")\
""",
    },

    # ---- Mode comparison ----
    {
        "cell_type": "markdown",
        "source": "### Step 12 — Mode comparison: keyword vs hybrid vs semantic",
    },
    {
        "cell_type": "code",
        "source": """\
# Side-by-side overall metrics
modes_present = sorted(results_df["mode"].unique())
overall = metrics_df[metrics_df["query_type"] == "ALL"].copy()

if len(overall) > 0:
    display(
        overall[[
            "mode", "n_queries",
            "recall@1", "recall@3", "recall@5",
            "precision@3", "mrr", "top1_accuracy", "ndcg@5"
        ]].style
        .background_gradient(
            subset=["recall@5", "precision@3", "mrr", "top1_accuracy", "ndcg@5"],
            cmap="RdYlGn", vmin=0, vmax=1
        )
        .set_caption("Overall metrics by search mode")
    )

# Bar chart comparing modes
import matplotlib.pyplot as plt

metric_cols = ["recall@5", "precision@3", "mrr", "top1_accuracy", "ndcg@5"]
plot_data = overall.set_index("mode")[metric_cols]

ax = plot_data.plot(kind="bar", figsize=(10, 4), ylim=(0, 1.1), rot=0, width=0.65)
ax.set_ylabel("Score")
ax.set_title("Phase 3 — Mode Comparison (overall)")
ax.grid(axis="y", linestyle=":", alpha=0.4)
plt.tight_layout()
plt.show()\
""",
    },

    # ---- Per-type comparison ----
    {
        "cell_type": "code",
        "source": """\
# Per-query-type: show recall@5 for each mode side by side
type_pivot = (
    metrics_df[metrics_df["query_type"] != "ALL"]
    .pivot(index="query_type", columns="mode", values="recall@5")
    .fillna(0)
    .sort_index()
)

if not type_pivot.empty:
    display(
        type_pivot.style
        .background_gradient(cmap="RdYlGn", vmin=0, vmax=1, axis=None)
        .format("{:.2f}")
        .set_caption("Recall@5 by query type and search mode")
    )

    ax = type_pivot.plot(kind="bar", figsize=(14, 5), ylim=(0, 1.1), rot=45, width=0.7)
    ax.set_ylabel("Recall@5")
    ax.set_title("Recall@5 by Query Type — Mode Comparison")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.show()\
""",
    },

    # ---- Hybrid top-1 view ----
    {
        "cell_type": "markdown",
        "source": "### Step 13 — Hybrid top-1 results with detected query type and alpha",
    },
    {
        "cell_type": "code",
        "source": """\
from src.search import detect_query_type, recommended_alpha

hybrid_top1 = results_df[
    (results_df["mode"] == "hybrid") & (results_df["rank"] == 1)
].copy()

if not hybrid_top1.empty:
    hybrid_top1["query_type_detected"] = hybrid_top1["query"].apply(detect_query_type)
    hybrid_top1["alpha"] = hybrid_top1["query"].apply(recommended_alpha)

    display(
        hybrid_top1[[
            "query_id", "query_type", "query_type_detected", "alpha",
            "query", "returned_source", "returned_page", "score",
            "doc_match", "keyword_coverage", "auto_relevance",
        ]].style
        .applymap(
            lambda v: "background-color:#d4edda" if v in ("relevant","likely_relevant")
            else ("background-color:#f8d7da" if v == "irrelevant" else ""),
            subset=["auto_relevance"]
        )
        .format({"alpha": "{:.2f}", "score": "{:.4f}", "keyword_coverage": "{:.2f}"})
        .set_caption("Hybrid top-1 results with adaptive alpha")
    )
else:
    print("No hybrid results yet — run Step 11 first.")\
""",
    },

    # ---- doc_pair_boost demo ----
    {
        "cell_type": "markdown",
        "source": """\
### Step 14 — Document-pair boost

`doc_pair_boost` is a post-processing utility that re-ranks any result list by
adding a small score bonus to results from the user's selected document pair.

This matters most for **global search** (all 10 PDFs), where a user who has
selected the Blood Inquiry pair should see Blood Inquiry results promoted ahead
of, say, a weakly-relevant Space Economy chunk with similar TF-IDF overlap.\
""",
    },
    {
        "cell_type": "code",
        "source": """\
from src.search import keyword_search, doc_pair_boost

demo_query = "compensation scheme redress"
demo_pair  = ["Volume_1-Blood-Inquiry-Recomm.pdf", "Volume_1-Blood-Inquiry-Response.pdf"]

raw_results   = keyword_search(demo_query, qa_corpus, top_k=8)
boosted_results = doc_pair_boost(raw_results, preferred_sources=demo_pair, boost=0.12)

raw_df = pd.DataFrame([
    {"rank": i, "source": r["source"].replace(".pdf",""),
     "page": r["page_number"], "score": round(r["score"], 4),
     "in_pair": r["source"] in demo_pair}
    for i, r in enumerate(raw_results, 1)
])
boosted_df = pd.DataFrame([
    {"rank": i, "source": r["source"].replace(".pdf",""),
     "page": r["page_number"], "score": round(r["score"], 4),
     "in_pair": r["source"] in demo_pair}
    for i, r in enumerate(boosted_results, 1)
])

print(f"Query: '{demo_query}'")
print(f"Preferred pair: Blood Inquiry (recomm + response)")
print()
print("Before boost:")
display(raw_df)
print("After +0.12 pair boost:")
display(boosted_df)\
""",
    },

    # ---- Phase 3 summary ----
    {
        "cell_type": "markdown",
        "source": """\
### Phase 3 Summary

| Improvement | Location | Effect |
|---|---|---|
| `detect_query_type` | `src/search.py` | Classifies queries into 5 types without ML |
| `recommended_alpha` | `src/search.py` | Returns blend weight matching query characteristics |
| `hybrid_search` | `src/search.py` | Combines TF-IDF and MiniLM with adaptive alpha |
| `_rec_heading_boost_scores` | `src/search.py` | Promotes numbered-recommendation chunks |
| `doc_pair_boost` | `src/search.py` | Post-processing boost for selected document pair |
| QA matrix hybrid mode | `src/qa_matrix.py` | `run_qa_matrix(modes=["keyword","semantic","hybrid"])` |

Hybrid mode targets the primary failure modes identified in the Phase 1 matrix:

- **paraphrase_miss**: semantic component (alpha < 1) bridges the vocabulary gap
- **ranking_issue**: combined score promotes correctly-matching chunks above competing chunks
- **wrong_doc_similar_topic**: document-pair boost counteracts cross-inquiry vocabulary overlap

The Phase 6 notebook will re-run the QA matrix after all phases and present a
before/after metric comparison as coursework evidence.\
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
            "id": f"p3_{abs(hash(src)) % 10**8:08x}",
            "metadata": {},
            "source": sl,
        }
    return {
        "cell_type": "code",
        "id": f"p3_{abs(hash(src)) % 10**8:08x}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": sl,
    }


def main() -> None:
    with open(nb_path, encoding="utf-8") as fh:
        nb = json.load(fh)

    existing_ids = {c.get("id", "") for c in nb["cells"]}
    first_id = f"p3_{abs(hash(CELLS[0]['source'])) % 10**8:08x}"
    if first_id in existing_ids:
        print("Phase 3 cells already present — nothing to do.")
        sys.exit(0)

    new_cells = [_make_cell(c) for c in CELLS]
    nb["cells"].extend(new_cells)

    with open(nb_path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)

    print(f"Injected {len(new_cells)} Phase 3 cells into {nb_path.name}")


if __name__ == "__main__":
    main()
