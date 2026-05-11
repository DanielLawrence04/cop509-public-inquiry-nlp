# COP509 Coursework: Policy Recommendation Analysis

This repository contains the COP509 Natural Language Processing coursework submission materials. The official marked deliverables are the two notebooks and their PDF exports; the web app is optional supporting evidence only.

## Official Submission Files

The final ZIP should contain:

- `COP509_Notebook1_Search.ipynb`
- `COP509_Notebook1_Search.pdf`
- `COP509_Notebook2_Extraction_Alignment.ipynb`
- `COP509_Notebook2_Extraction_Alignment.pdf`

## Repository Scope

- Full available preset corpus: 8 document pairs / 16 PDFs in `data/raw/`.
- Task 1 search evaluation corpus: 5 coursework-given pairs / 10 PDFs, evaluated with `data/ground_truth/qa_matrix_queries.json`.
- Task 2 final output: 246 recommendation rows across the full loaded/extended corpus.

The final validated evidence files are:

- `outputs/final_recommendations_246.json`
- `outputs/evaluation_predictions.csv`

## Setup

Create and activate an environment, then install notebook dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Tesseract OCR is optional. If it is unavailable, the notebooks still run and report OCR usage in the extraction audit.

## Run The Notebooks

Launch Jupyter from the project root:

```powershell
jupyter lab
```

Run in this order:

1. `notebooks/COP509_Notebook1_Search.ipynb`
2. `notebooks/COP509_Notebook2_Extraction_Alignment.ipynb`

Both notebooks detect the project root automatically and use relative paths for `data/`, `src/`, and `outputs/`.

## Export PDFs

From the project root:

```powershell
jupyter nbconvert --to pdf notebooks/COP509_Notebook1_Search.ipynb
jupyter nbconvert --to pdf notebooks/COP509_Notebook2_Extraction_Alignment.ipynb
```

If local PDF export fails because TeX is unavailable, export through JupyterLab's browser print/PDF flow or another environment with a working PDF toolchain.

## Validation

Validate the final Task 2 export:

```powershell
python scripts/validate_recommendation_export.py --baseline outputs/final_recommendations_246.json
```

Expected final counts:

- total rows: 246
- document-pair counts: 33, 19, 40, 10, 58, 58, 19, 9
- classification distribution: accepted 104, partial 139, rejected 1, not_addressed 2

Check for accidental local paths before commit:

```powershell
rg "C:\\Users|Downloa[d]s|Path[.]home" notebooks README.md src backend scripts
```

## Data And Outputs

Required notebook inputs:

- `data/raw/*.pdf`
- `data/ground_truth/qa_matrix_queries.json`
- source modules in `src/`
- preset map in `backend/core/presets.py`
- validator in `scripts/validate_recommendation_export.py`

`data/ground_truth/labels.json` is not a 246-row manual benchmark. Notebook 2 therefore reports prediction-only evidence unless a full manual label file is added later.

## Optional Web App

The optional web app lives in `frontend/` and `backend/`. It is supporting evidence only and is not required to mark the notebooks.

Current deployment note: the frontend currently points to a local FastAPI backend at `http://localhost:8000`. A Vercel-hosted frontend will therefore need either a separately hosted backend API or a later deployment configuration pass. This repository pass does not modify frontend/backend deployment code.

Local app commands:

```powershell
# backend
python -m uvicorn backend.main:app --reload --port 8000

# frontend
cd frontend
npm run dev
```

GitHub link: add after push.

Vercel link: add after optional deployment.

## Repository Hygiene

The `.gitignore` excludes secrets, virtual environments, Python/Jupyter caches, logs, local archives, `frontend/node_modules/`, `frontend/dist/`, runtime output caches, and old timestamped exports.

It keeps the notebooks, source modules, required data, README, requirements, and canonical final output evidence.

## Limitations

- The search evaluation uses a transparent 50-query automatic relevance matrix rather than a large human-labelled passage benchmark.
- Full 246-row manual ground truth is unavailable, so Task 2 reports prediction-only validation evidence.
- Widget cells are included for interactive use; static fallback tables are included so PDF exports remain readable.
- The web app is optional supporting evidence and should not be treated as the official marked deliverable.
