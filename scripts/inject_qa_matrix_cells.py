"""
One-time script: append the Search QA Matrix section to
COP509_Notebook1_Search.ipynb.

Run from the project root:
    python scripts/inject_qa_matrix_cells.py
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
nb_path = project_root / "notebooks" / "COP509_Notebook1_Search.ipynb"

# ---------------------------------------------------------------------------
# Cell definitions  (markdown + code pairs)
# ---------------------------------------------------------------------------

CELLS = [
    # ---- Section header ----
    {
        "cell_type": "markdown",
        "source": """---
## 10. Search QA Matrix — Extended Evaluation

The sections above demonstrate a working TF-IDF retriever and measure it against
six gold-standard anchored queries.  This section extends that evaluation with a
**Search QA Matrix**: a structured test harness that runs a wider, more
representative set of queries across *all ten* documents in the collection
(both recommendation and response PDFs), records every search result in a
long-format table, automatically labels result relevance, and computes a full
suite of retrieval metrics.

### Why this evaluation design is fair and useful

| Design choice | Justification |
|---|---|
| All ten PDFs included | Response documents are real retrieval targets; excluding them understates cross-document errors. |
| Same `chunk_pages` pipeline | Evaluation reflects the live search system — not a separate preprocessor. |
| 50 queries across 10 types | Covers exact phrases, paraphrases, named entities, numeric queries, and ambiguous queries so failure modes surface by category. |
| Auto-labelling (doc_match + keyword_coverage + anchor_match) | Provides immediate signal without requiring 750 manual labels; blank `manual_relevance` column allows human override. |
| Metrics: Recall@k, Precision@k, MRR, Top-1, Doc Acc, nDCG@5 | Together they capture coverage, precision, and ranking quality at multiple cut-offs. |
| Error categorisation | Groups failures by root cause (chunk_too_small, paraphrase_miss, ranking_issue, etc.) so fixes can be prioritised by impact. |

The matrix is designed to be **re-run after each improvement phase** so that
before/after metric comparisons provide direct evidence of progress.""",
    },
    # ---- Imports ----
    {
        "cell_type": "code",
        "source": """# QA Matrix imports
from src.qa_matrix import (
    run_qa_matrix,
    load_query_bank,
    queries_to_dataframe,
    top1_results_table,
    failures_table,
    discover_pdfs,
    build_qa_corpus,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
)
from src.qa_matrix_report import (
    error_analysis_table,
    query_type_error_breakdown,
    generate_findings_report,
)

# Paths (derived from project_root set in cell 0 — no hardcoded absolutes)
QA_QUERY_PATH = project_root / "data" / "ground_truth" / "qa_matrix_queries.json"
ALL_PDFS = discover_pdfs(DATA_DIR)

print(f"Query bank  : {QA_QUERY_PATH.name}")
print(f"PDFs found  : {len(ALL_PDFS)}")
for p in ALL_PDFS:
    print(f"  {p.name}")""",
    },
    # ---- Corpus ----
    {
        "cell_type": "markdown",
        "source": """### Step 1 — Build the all-documents corpus

Unlike the demo in Section 2 (which loads one PDF) and the evaluation corpus
in Section 5 (which loads only recommendation PDFs), the QA Matrix corpus
includes **all ten documents**.  This is intentional: retrieval queries in a
real search box are not restricted to recommendation files, and response
documents contain complementary passages that a user might legitimately want
to find.

The same `chunk_pages(chunk_size=200, overlap=30)` pipeline is used so
evaluation results are directly comparable to live search behaviour.""",
    },
    {
        "cell_type": "code",
        "source": """# Build the unified QA corpus (all 10 PDFs)
# Re-use CHUNK_SIZE / OVERLAP constants set in Section 2
qa_corpus = build_qa_corpus(ALL_PDFS, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

corpus_summary = {}
for chunk in qa_corpus:
    corpus_summary.setdefault(chunk["source"], 0)
    corpus_summary[chunk["source"]] += 1

corpus_info_df = pd.DataFrame([
    {"document": src, "chunks": n}
    for src, n in sorted(corpus_summary.items())
])
corpus_info_df["doc_type"] = corpus_info_df["document"].apply(
    lambda x: "response" if "response" in x.lower() else "recommendation"
)

print(f"Total chunks : {len(qa_corpus)}")
print(f"Documents    : {len(corpus_info_df)}")
display(corpus_info_df)""",
    },
    # ---- Query bank ----
    {
        "cell_type": "markdown",
        "source": """### Step 2 — Load and inspect the query bank

The query bank (`data/ground_truth/qa_matrix_queries.json`) contains
**50 queries across 10 categories**.  Each query records:

- the query text and type
- expected source documents (`expected_docs`)
- expected keywords and an optional anchor substring for auto-labelling
- search modes to test (keyword, semantic, or both)

The categories are designed to stress-test different failure modes:

| Category | Purpose |
|---|---|
| `exact_phrase` | Confirms basic lexical matching works for known phrases |
| `paraphrase_semantic` | Tests recall when query vocabulary differs from passage |
| `recommendation_specific` | Targets numbered recommendation headings |
| `response_specific` | Tests cross-document retrieval into response files |
| `entity_person_org` | Targets named entity-heavy passages |
| `policy_topic` | Broad topic retrieval without specific phrase |
| `numeric_date` | Tests passages containing statistics or dates |
| `recomm_response_gap` | Query phrased in one document's vocabulary; gold text in the other |
| `broad_exploratory` | Long multi-concept queries mimicking real user behaviour |
| `ambiguous` | Short/vague queries that could validly match multiple documents |""",
    },
    {
        "cell_type": "code",
        "source": """# Load query bank and display summary
queries_df = queries_to_dataframe(load_query_bank(QA_QUERY_PATH))

print(f"Total queries : {len(queries_df)}")
print("\\nQueries by type:")
print(queries_df["query_type"].value_counts().to_string())
print()
display(
    queries_df[["query_id", "query_type", "query", "expected_docs", "has_anchor"]]
    .head(20)
    .style.set_caption("Query bank (first 20 of 50)")
)""",
    },
    # ---- Run matrix ----
    {
        "cell_type": "markdown",
        "source": """### Step 3 — Run the evaluation matrix (keyword mode)

`run_qa_matrix` executes every query in keyword mode, auto-labels each result,
and returns three DataFrames.  This cell runs keyword search only; semantic
search is added in Step 7 if `sentence-transformers` is available.

**Auto-labelling rules**

| Label | Condition |
|---|---|
| `relevant` | Chunk contains the anchor substring (normalised) |
| `likely_relevant` | `doc_match=True` AND ≥50% of expected keywords present |
| `partial` | `doc_match=True` AND 25–49% of expected keywords present |
| `doc_match_only` | `doc_match=True` but keyword coverage < 25% |
| `irrelevant` | `doc_match=False` |
| `unknown` | No `expected_docs` specified (ambiguous queries) |

The `manual_relevance` column is left blank for human grading.""",
    },
    {
        "cell_type": "code",
        "source": """# Run QA matrix — keyword mode
# Pass prebuilt_corpus to avoid re-loading PDFs
queries_df_out, results_df, metrics_df = run_qa_matrix(
    data_dir=DATA_DIR,
    query_bank_path=QA_QUERY_PATH,
    chunk_size=CHUNK_SIZE,
    overlap=OVERLAP,
    top_k=5,
    modes=["keyword"],
    prebuilt_corpus=qa_corpus,
)

print(f"Results rows : {len(results_df)}")
print(f"Unique queries evaluated : {results_df['query_id'].nunique()}")
print(f"Modes run    : {results_df['mode'].unique().tolist()}")
print("\\nAuto-relevance distribution (all ranks):")
print(results_df["auto_relevance"].value_counts().to_string())""",
    },
    # ---- Top-1 results ----
    {
        "cell_type": "markdown",
        "source": """### Step 4 — Top-1 results per query

The table below shows the rank-1 result for every query in keyword mode.
The `auto_relevance` column and `error_category` (for failures) give an
immediate per-query assessment without requiring manual inspection of all
five results per query.""",
    },
    {
        "cell_type": "code",
        "source": """# Compact top-1 view
t1 = top1_results_table(results_df)
display(
    t1[[
        "query_id", "query_type", "query",
        "returned_source", "returned_page", "score",
        "doc_match", "keyword_coverage", "anchor_match",
        "auto_relevance", "error_category",
    ]].style.set_caption("Top-1 result per query (keyword mode)")
    .applymap(
        lambda v: "background-color: #d4edda" if v in ("relevant", "likely_relevant")
        else ("background-color: #f8d7da" if v == "irrelevant"
              else ""),
        subset=["auto_relevance"]
    )
)""",
    },
    # ---- Metrics ----
    {
        "cell_type": "markdown",
        "source": """### Step 5 — Retrieval metrics by query type

The metrics table aggregates retrieval quality per search mode and query category.

| Metric | What it measures |
|---|---|
| Recall@k | Fraction of queries where ≥1 relevant result appears in top-k |
| Precision@k | Mean fraction of top-k results that are relevant |
| MRR | Mean Reciprocal Rank — how early does the first relevant result appear? |
| Top-1 accuracy | Fraction of queries where rank-1 is relevant |
| Doc accuracy | Fraction of queries where rank-1 source is an expected document |
| nDCG@5 | Normalised Discounted Cumulative Gain — grades partial/full relevance |

*Relevance is binary (relevant + likely_relevant = hit) for Recall/Precision/MRR/Top-1.
nDCG uses graded labels: relevant=3, likely_relevant=2, partial/doc_match_only=1, irrelevant=0.*""",
    },
    {
        "cell_type": "code",
        "source": """# Metrics overview
overall_row = metrics_df[metrics_df["query_type"] == "ALL"]
type_rows   = metrics_df[metrics_df["query_type"] != "ALL"].sort_values(
    ["mode", "recall@5"], ascending=[True, False]
)

print("=== OVERALL ===")
display(overall_row[[
    "mode", "n_queries", "recall@1", "recall@3", "recall@5",
    "precision@3", "mrr", "top1_accuracy", "doc_accuracy", "ndcg@5"
]])

print("\\n=== BY QUERY TYPE ===")
display(
    type_rows[[
        "mode", "query_type", "n_queries",
        "recall@5", "precision@3", "mrr", "top1_accuracy", "ndcg@5"
    ]].style.background_gradient(
        subset=["recall@5", "precision@3", "mrr"], cmap="RdYlGn", vmin=0, vmax=1
    ).set_caption("Metrics by query type (keyword mode)")
)""",
    },
    {
        "cell_type": "code",
        "source": """# Bar chart: recall@5, precision@3, MRR by query type (keyword)
import matplotlib.pyplot as plt

kw_types = metrics_df[
    (metrics_df["mode"] == "keyword") & (metrics_df["query_type"] != "ALL")
].set_index("query_type")[["recall@5", "precision@3", "mrr"]]

ax = kw_types.plot(
    kind="bar", figsize=(14, 5), ylim=(0, 1.05), rot=45
)
ax.set_ylabel("Score")
ax.set_title("QA Matrix — Keyword Search Metrics by Query Type")
ax.grid(axis="y", linestyle=":", alpha=0.4)
plt.tight_layout()
plt.show()""",
    },
    # ---- Error analysis ----
    {
        "cell_type": "markdown",
        "source": """### Step 6 — Error analysis

The error analysis table counts **top-1 failures** (queries where rank-1 is not
relevant or likely-relevant) and groups them by root cause.

Error categories are assigned automatically using a decision tree:

| Category | Decision rule |
|---|---|
| `chunk_too_small` | Returned chunk < 60 words |
| `chunk_too_large` | Returned chunk > 450 words |
| `wrong_doc_similar_topic` | Wrong source and TF-IDF score > 0.15 (vocabulary overlap) |
| `wrong_document` | Wrong source and low score |
| `paraphrase_miss` | Correct document but keyword coverage < 20% |
| `ranking_issue` | Correct document with keyword coverage ≥ 20% but ranked below 1 |
| `keyword_coverage_low` | Correct document, partial keyword coverage |

The cross-tabulation below shows which query types suffer from which error types.""",
    },
    {
        "cell_type": "code",
        "source": """# Error summary table
error_df = error_analysis_table(results_df)

if error_df.empty:
    print("No top-1 failures detected.")
else:
    n_fail = error_df["count"].sum()
    n_total = results_df["query_id"].nunique()
    print(f"Top-1 failures : {n_fail} of {n_total} queries ({100*n_fail/n_total:.1f}%)")
    print()
    display(error_df.style.set_caption("Top-1 failures by error category"))

# Cross-tabulation: query_type × error_category
breakdown = query_type_error_breakdown(results_df)
if not breakdown.empty:
    print("\\nFailure breakdown by query type:")
    display(breakdown)""",
    },
    {
        "cell_type": "code",
        "source": """# Inspect the actual failure rows
fail_rows = failures_table(results_df)
if fail_rows.empty:
    print("No top-1 failures to display.")
else:
    print(f"Showing {len(fail_rows)} top-1 failure(s):")
    display(
        fail_rows[[
            "query_id", "query_type", "mode", "query",
            "returned_source", "returned_page", "score",
            "doc_match", "keyword_coverage", "auto_relevance",
            "error_category", "returned_preview",
        ]]
    )""",
    },
    # ---- Semantic search ----
    {
        "cell_type": "markdown",
        "source": """### Step 7 — Semantic search evaluation (optional)

The semantic search mode uses `all-MiniLM-L6-v2` (sentence-transformers) to
encode queries and chunks as dense vectors, then ranks by cosine similarity
with a hybrid lexical rerank (see `src/search.py:semantic_search`).

This cell appends semantic results to `results_df` and `metrics_df` so the
two modes can be compared side-by-side.  If `sentence-transformers` is not
installed the cell prints a warning and continues without semantic results.

In Google Colab: `!pip install sentence-transformers` then re-run this cell.""",
    },
    {
        "cell_type": "code",
        "source": """# Run semantic mode (extend existing results_df and metrics_df)
try:
    from src.search import build_embeddings, semantic_search

    print("Building embeddings…")
    qa_embeddings = build_embeddings(qa_corpus)
    print(f"Embeddings: {qa_embeddings.shape}")

    _, results_sem, metrics_sem = run_qa_matrix(
        data_dir=DATA_DIR,
        query_bank_path=QA_QUERY_PATH,
        chunk_size=CHUNK_SIZE,
        overlap=OVERLAP,
        top_k=5,
        modes=["semantic"],
        prebuilt_corpus=qa_corpus,
        prebuilt_embeddings=qa_embeddings,
    )

    # Merge with keyword results
    results_df = pd.concat([results_df, results_sem], ignore_index=True)
    metrics_df = pd.concat([metrics_df, metrics_sem], ignore_index=True)

    print("\\nSemanticoverall metrics:")
    sem_overall = metrics_sem[metrics_sem["query_type"] == "ALL"]
    display(sem_overall[[
        "mode", "n_queries", "recall@1", "recall@5", "precision@3",
        "mrr", "top1_accuracy", "ndcg@5"
    ]])

    # Side-by-side comparison
    modes_present = results_df["mode"].unique().tolist()
    if len(modes_present) > 1:
        compare = metrics_df[metrics_df["query_type"] == "ALL"][[
            "mode", "recall@5", "precision@3", "mrr", "top1_accuracy", "ndcg@5"
        ]]
        print("\\nMode comparison (overall):")
        display(compare)

except ImportError:
    print("sentence-transformers not installed — semantic mode skipped.")
    print("Install with: pip install sentence-transformers")
except Exception as exc:
    print(f"Semantic mode error: {exc}")""",
    },
    # ---- Findings report ----
    {
        "cell_type": "markdown",
        "source": """### Step 8 — Findings report

The findings report summarises the evaluation results in plain language:
what worked, what failed, the most common failure types, and a prioritised
list of improvements with evidence from the metric and error tables.

This report is auto-generated from the DataFrames above and should be
re-run after each improvement phase to reflect updated performance.""",
    },
    {
        "cell_type": "code",
        "source": """# Generate and display findings report
error_df = error_analysis_table(results_df[results_df["mode"] == "keyword"])
report_md = generate_findings_report(results_df, metrics_df, error_df)

from IPython.display import Markdown
display(Markdown(report_md))""",
    },
    # ---- Manual relevance ----
    {
        "cell_type": "markdown",
        "source": """### Step 9 — Manual relevance labelling guide

The `manual_relevance` column in `results_df` is blank.  To refine the
auto-labels, fill it with one of three values:

| Label | Meaning |
|---|---|
| `relevant` | The passage directly answers the query |
| `partial` | The passage is topically related but does not directly answer |
| `irrelevant` | The passage does not address the query |

Run the cell below to export the top-1 results to a CSV for offline labelling.
After labelling, reload the CSV and recompute metrics using the `manual_relevance`
column instead of `auto_relevance`.""",
    },
    {
        "cell_type": "code",
        "source": """# Export top-1 results for manual labelling
labelling_export_path = project_root / "outputs" / "qa_matrix_label_sheet.csv"
labelling_export_path.parent.mkdir(parents=True, exist_ok=True)

export_cols = [
    "query_id", "query_type", "mode", "query",
    "returned_source", "returned_page", "score",
    "doc_match", "keyword_coverage", "anchor_match", "auto_relevance",
    "manual_relevance", "error_category", "returned_preview",
]
top1 = results_df[results_df["rank"] == 1][export_cols]
top1.to_csv(labelling_export_path, index=False, encoding="utf-8")
print(f"Label sheet saved: {labelling_export_path}")
print(f"Rows to label: {len(top1)}")
print(
    "Fill the 'manual_relevance' column with: relevant / partial / irrelevant, "
    "then reload and rerun metrics."
)

# Preview
display(top1[["query_id", "query_type", "query", "auto_relevance", "manual_relevance"]].head(10))""",
    },
]


# ---------------------------------------------------------------------------
# Load, extend, and save the notebook
# ---------------------------------------------------------------------------

def build_nb_cell(cell_def: dict) -> dict:
    """Convert a simple cell definition into a valid nbformat v4 cell dict."""
    src = cell_def["source"]
    source_lines = [line + "\n" for line in src.split("\n")]
    if source_lines:
        source_lines[-1] = source_lines[-1].rstrip("\n")  # no trailing newline on last line

    if cell_def["cell_type"] == "markdown":
        return {
            "cell_type": "markdown",
            "id": f"qa_matrix_{abs(hash(src)) % 10**8:08x}",
            "metadata": {},
            "source": source_lines,
        }
    else:
        return {
            "cell_type": "code",
            "id": f"qa_matrix_{abs(hash(src)) % 10**8:08x}",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source_lines,
        }


def main() -> None:
    with open(nb_path, encoding="utf-8") as fh:
        nb = json.load(fh)

    # Guard: don't double-inject
    existing_ids = {c.get("id", "") for c in nb.get("cells", [])}
    first_new_id = f"qa_matrix_{abs(hash(CELLS[0]['source'])) % 10**8:08x}"
    if first_new_id in existing_ids:
        print("QA Matrix cells already present — nothing to do.")
        sys.exit(0)

    new_cells = [build_nb_cell(c) for c in CELLS]
    nb["cells"].extend(new_cells)

    with open(nb_path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)

    print(f"Injected {len(new_cells)} cells into {nb_path.name}")


if __name__ == "__main__":
    main()
