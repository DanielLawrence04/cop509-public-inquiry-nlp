"""
DEPRECATED — do NOT run this script.

This was a one-shot migration script used during Phase 2 to wire up the global
search corpus.  The notebook has since advanced to Phase 5/6 (advanced_explorer,
chunk_pages_v2, hybrid search).  Running this script NOW would DOWNGRADE cells
43febd5b and f29d8759 back to the old Phase 2 keyword-only retrieval_explorer
widget, overwriting the current Phase 5 implementation.

retrieval_explorer.py has been deleted.  This script is kept only as a build-
history record and must not be executed.
"""
import sys
print("ERROR: patch_phase2_cells.py is deprecated and must not be run. See module docstring.", file=sys.stderr)
sys.exit(1)

import json
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
nb_path = project_root / "notebooks" / "COP509_Notebook1_Search.ipynb"

MARKDOWN_4_ID = "43febd5b"
WIDGET_CELL_ID = "f29d8759"

NEW_MARKDOWN_4 = """\
---
## 4. Interactive Search Demo

The widget below provides a real-time search interface across **all documents**
in the collection (both recommendation and response PDFs).

**Phase 2 correction:** the original cell searched only the first loaded PDF.
The widget now builds a unified corpus from all available PDFs using the same
`chunk_pages` pipeline as the live app, giving the interactive demo the same
retrieval scope as the QA Matrix evaluation in Section 10.

Features:
- Free-text query input with live results
- `top-k` slider (1–20 results)
- Document filter dropdown to restrict search to a single file
- Result cards with document badge, page number, score bar, and term highlighting\
"""

NEW_WIDGET_CELL = """\
# Phase 2: build a global corpus for the interactive demo (all PDFs, same pipeline)
from src.qa_matrix import build_qa_corpus, discover_pdfs as _discover_pdfs
from src.widgets.retrieval_explorer import show as show_retrieval_explorer

_all_pdfs = _discover_pdfs(DATA_DIR)
all_search_chunks = build_qa_corpus(_all_pdfs, chunk_size=CHUNK_SIZE, overlap=OVERLAP)

print(f"Interactive corpus: {len(all_search_chunks):,} chunks across {len(_all_pdfs)} documents")
show_retrieval_explorer(all_search_chunks, keyword_search)\
"""


def _to_source_lines(text: str) -> list[str]:
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def main() -> None:
    with open(nb_path, encoding="utf-8") as fh:
        nb = json.load(fh)

    patched = 0
    for cell in nb["cells"]:
        cid = cell.get("id", "")
        if cid == MARKDOWN_4_ID:
            cell["source"] = _to_source_lines(NEW_MARKDOWN_4)
            patched += 1
            print(f"  Patched markdown cell {cid}")
        elif cid == WIDGET_CELL_ID:
            cell["source"] = _to_source_lines(NEW_WIDGET_CELL)
            cell["outputs"] = []
            cell["execution_count"] = None
            patched += 1
            print(f"  Patched code cell {cid}")

    if patched != 2:
        print(f"WARNING: expected 2 patches but applied {patched}")

    with open(nb_path, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)

    print(f"Saved {nb_path.name}  ({patched} cell(s) patched)")


if __name__ == "__main__":
    main()
