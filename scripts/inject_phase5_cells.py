"""
Phase 5 notebook injection: append Section 13 (Advanced Widget) cells.

Run from the project root:
    python scripts/inject_phase5_cells.py
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
## 13. Phase 5 — Advanced Search Widget

The Phase 5 widget replaces the baseline `retrieval_explorer` with a richer
interactive interface.  All new features are wired to the Phase 3–4 functions
already in `src/`:

| Feature | Implementation |
|---|---|
| **Mode toggle** | Keyword / Semantic / Hybrid via `ToggleButtons` |
| **Confidence badge** | High ≥ 0.50 · Medium ≥ 0.20 · Low < 0.20 |
| **Heading badge** | `detect_chunk_heading` — shows section title if present |
| **Why matched row** | Matched terms · query type · adaptive α (hybrid only) |
| **Expandable context** | `expand_with_context(window=1)` behind HTML5 `<details>` |

Semantic and hybrid modes require the pre-built `qa_embeddings` array from
Step 7.  If embeddings are not available the widget falls back to keyword-only
and shows a status message.\
""",
    },

    {
        "cell_type": "markdown",
        "source": "### Step 20 — Launch advanced widget (keyword mode)",
    },
    {
        "cell_type": "code",
        "source": """\
from src.widgets.advanced_explorer import show_advanced

# Keyword-only: corpus from Phase 4 v2 build (richer chunks)
print(f"V2 corpus: {len(qa_corpus_v2):,} chunks across {len(set(c['source'] for c in qa_corpus_v2))} documents")
show_advanced(qa_corpus_v2)\
""",
    },

    {
        "cell_type": "markdown",
        "source": "### Step 21 — Launch advanced widget (all modes — requires embeddings)",
    },
    {
        "cell_type": "code",
        "source": """\
# Pass qa_embeddings_v2 to enable semantic and hybrid mode buttons.
# If qa_embeddings_v2 is not yet built, run Step 17 first.
try:
    _ = qa_embeddings_v2
    print(f"Embeddings found: {qa_embeddings_v2.shape} — all three modes available")
    show_advanced(qa_corpus_v2, embeddings=qa_embeddings_v2)
except NameError:
    print("qa_embeddings_v2 not available — showing keyword-only widget.")
    print("Run Step 17 first to build v2 embeddings, then re-run this cell.")
    show_advanced(qa_corpus_v2)\
""",
    },

    {
        "cell_type": "markdown",
        "source": """\
### Step 22 — Heading detection spot-check

`detect_chunk_heading` is a display-only utility that extracts a leading
section heading from a chunk.  It fires on numbered-recommendation lines,
ALL-CAPS section titles, and `Recommendation N:` patterns.\
""",
    },
    {
        "cell_type": "code",
        "source": """\
from src.chunking import detect_chunk_heading

# Sample the first 300 chunks and show any with a detected heading
heading_hits = [
    (c["source"].replace(".pdf",""), c["page_number"],
     detect_chunk_heading(c["text"]), c["text"][:80])
    for c in qa_corpus_v2[:300]
    if detect_chunk_heading(c["text"])
]

heading_df = pd.DataFrame(heading_hits, columns=["source", "page", "heading", "text_start"])
print(f"Headings detected in first 300 chunks: {len(heading_df)}")
if not heading_df.empty:
    display(heading_df.style.set_caption("Detected chunk headings (sample)"))\
""",
    },

    {
        "cell_type": "markdown",
        "source": """\
### Phase 5 Summary

The advanced widget exposes all Phase 3–4 capabilities in a single interface:

- **Mode toggle** lets the user switch between retrieval strategies without
  writing code, making the performance differences immediately observable.
- **Confidence badge** provides an at-a-glance quality signal so the user knows
  whether a top result is a strong match or a weak best-effort result.
- **Heading badge** surfaces section context for numbered recommendations,
  helping users identify whether a result is from the recommendation text or
  a supporting paragraph.
- **Why matched** explains the retrieval signal, increasing interpretability —
  a key requirement for human-in-the-loop policy search tools.
- **Expandable context** lets users read surrounding text without navigating to
  the source PDF, reducing cognitive switching cost.

Phase 6 (next section) re-runs the full QA matrix across all modes and
chunking strategies to quantify the cumulative improvement from Phases 2–5.\
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
            "id": f"p5_{abs(hash(src)) % 10**8:08x}",
            "metadata": {},
            "source": sl,
        }
    return {
        "cell_type": "code",
        "id": f"p5_{abs(hash(src)) % 10**8:08x}",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": sl,
    }


def main() -> None:
    with open(nb_path, encoding="utf-8") as fh:
        nb = json.load(fh)

    existing_ids = {c.get("id", "") for c in nb["cells"]}
    first_id = f"p5_{abs(hash(CELLS[0]['source'])) % 10**8:08x}"
    if first_id in existing_ids:
        print("Phase 5 cells already present — nothing to do.")
        sys.exit(0)

    new_cells = [_make_cell(c) for c in CELLS]
    nb["cells"].extend(new_cells)

    with open(nb_path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)

    print(f"Injected {len(new_cells)} Phase 5 cells into {nb_path.name}")


if __name__ == "__main__":
    main()
