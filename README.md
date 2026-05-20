# COP509 Coursework: Public Inquiry NLP System

This repository is the final local-first submission package for the COP509 Natural Language Processing coursework. The marked notebooks and PDF exports are the official deliverables, and the repository also includes supporting evidence for the fuller system: semantic/hybrid search, a local web app, screenshots, final extraction/alignment exports and automated QA tests.

GitHub repository: <https://github.com/DanielLawrence04/cop509-public-inquiry-nlp>

## Official Marked Deliverables

- `COP509_Notebook1_Search.ipynb`
- `COP509_Notebook1_Search.pdf`
- `COP509_Notebook2_Extraction_Alignment.ipynb`
- `COP509_Notebook2_Extraction_Alignment.pdf`

Notebook 1 covers Task 1 search. Notebook 2 covers Task 2a recommendation extraction, Task 2b response matching/classification and Task 2c evaluation evidence.

## Open in Google Colab

Notebook 1 - Task 1 Search System:  
https://colab.research.google.com/github/DanielLawrence04/cop509-public-inquiry-nlp/blob/main/COP509_Notebook1_Search.ipynb

Notebook 2 - Task 2 Extraction, Alignment and Classification:  
https://colab.research.google.com/github/DanielLawrence04/cop509-public-inquiry-nlp/blob/main/COP509_Notebook2_Extraction_Alignment.ipynb

In Colab, run the setup cells at the top first. These clone the full GitHub repository and install the required packages so the notebooks can access `src/`, `data/`, `outputs/` and the PDF files.

The semantic/hybrid search cells are optional showcase cells. If Colab cannot download the MiniLM model from Hugging Face, the notebook automatically falls back to the reproducible TF-IDF baseline used for marking.

## Standard Marking Run

The standard marking path uses the lightweight, reproducible dependencies in `requirements.txt`. Notebook 1 defaults to TF-IDF lexical retrieval so it runs without large model downloads. Notebook 2 reads the final validated exports and does not overwrite source data.

Install the default dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

Launch Jupyter:

```powershell
jupyter lab
```

Run in this order:

1. `COP509_Notebook1_Search.ipynb`
2. `COP509_Notebook2_Extraction_Alignment.ipynb`

Both notebooks detect the project root automatically and use relative paths for `data/`, `src/` and `outputs/`.

## Full Showcase Mode

The full showcase mode demonstrates the extra engineering work beyond the reliable baseline:

- semantic search with MiniLM sentence embeddings;
- hybrid search combining lexical TF-IDF and semantic evidence;
- the optional local FastAPI/Vite web app;
- screenshots of the local app;
- final JSON/CSV extraction and alignment exports;
- automated QA/regression tests;
- additional supporting inquiry documents used by the fuller 8-pair pipeline.

Install this for full semantic/hybrid search mode:

```powershell
python -m pip install -r requirements-optional.txt
```

Then re-run the Notebook 1 section `Full showcase mode: semantic and hybrid search`. If the optional dependencies and MiniLM model are available, the notebook evaluates `keyword`, `semantic` and `hybrid` modes using the same QA matrix and saves:

- `outputs/search_evaluation_comparison.csv`

If optional dependencies or the model are unavailable, the cell prints a clear setup message and leaves the standard TF-IDF marking results unchanged.

## Project Layout

- `data/raw/`: source inquiry/report and government response PDFs.
- `data/ground_truth/`: query banks and validation support files.
- `src/`: shared NLP pipeline modules used by notebooks and app.
- `backend/`: optional local FastAPI backend for the supporting web app.
- `frontend/`: optional Vite frontend for inspecting the pipeline.
- `outputs/`: final evidence exports and optional generated comparison tables.
- `screenshots/`: local web-app screenshots used as supporting evidence.
- `tests/`: automated QA/regression tests used during development.

Canonical final Task 2 evidence:

- `outputs/final_recommendations_246.json`
- `outputs/evaluation_predictions.csv`

Expected final Task 2 counts:

- total rows: 246
- document-pair counts: 33, 19, 40, 10, 58, 58, 19, 9
- classification distribution: accepted 104, partial 139, rejected 1, not_addressed 2

## Optional Local Web App

The web app is supporting evidence, not the official marked deliverable. It demonstrates the same local workflow interactively: document loading, hybrid passage search, recommendation analysis and evaluation summaries.

Start the backend from the repository root after installing backend dependencies:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

Start the frontend in a second shell:

```powershell
cd frontend
npm install
npm run dev
```

Open the local Vite URL shown in the terminal. The frontend calls `http://localhost:8000` by default. To override it locally, set `VITE_API_BASE` as shown in `frontend/.env.example`.

The final submission is local-first because the full corpus, optional model setup and pipeline outputs are better suited to reliable local marking than limited free hosting. This is a deployment/resource choice, not a core system failure.

## Screenshots And QA Evidence

Screenshots in `screenshots/` show the optional app evidence:

- `01_documents_tab_8pairs_loaded.png`
- `02_search_tab_hybrid_retrieval.png`
- `03_recommendation_analysis_tab.png`
- `04_evaluation_summary_246_classified.png`

The `tests/` folder documents automated QA coverage for recommendation extraction, response-unit extraction, response matching/classification, final export validation and frontend evaluation-summary behaviour. See `tests/README.md`.

## Rubric Mapping

| Rubric area                              | Where covered                                                                                                                               |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Task 1 search system                     | `COP509_Notebook1_Search.ipynb`, sections on extraction, chunking, TF-IDF retrieval, semantic/hybrid showcase mode and QA matrix evaluation |
| Task 2a recommendation extraction        | `COP509_Notebook2_Extraction_Alignment.ipynb`, final export summary, live pair demonstration and extraction examples                        |
| Task 2b response matching/classification | Notebook 2 alignment/classification sections, confidence summaries and hard-case tables                                                     |
| Task 2c evaluation                       | Notebook 2 validation checks and expected-vs-actual sample evaluation                                                                       |
| Interactive demos                        | Notebook widgets plus optional local FastAPI/Vite app                                                                                       |
| Evaluation                               | Notebook 1 ranked retrieval metrics; Notebook 2 export validation, label distributions, confidence summaries and sample comparison          |
| Challenges                               | Challenge/resolution sections in both notebooks                                                                                             |
| Documentation                            | This README, tutorial-style notebook narrative, `tests/README.md` and screenshot/export evidence                                            |

## Limitations

- The standard Notebook 1 run reports TF-IDF baseline metrics by default; semantic/hybrid metrics are generated only when full showcase mode is run locally.
- The Task 1 search evaluation uses a transparent 50-query automatic relevance matrix rather than a large human-labelled passage benchmark.
- A full 246-row manual ground-truth label file is unavailable, so Notebook 2 avoids claiming full-corpus accuracy/F1.
- Widget cells are included for interactive notebook use; static fallback tables are included so PDF exports remain readable.
