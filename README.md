# COP509 Coursework: Local Submission Package

This repository is the final local-first submission package for the COP509 Natural Language Processing coursework. The notebooks and their PDF exports are the source of truth for marking. The web app is optional supporting evidence only.

Everything needed for assessment runs locally; no cloud account or external API URL is required.

## Marked Deliverables

- `COP509_Notebook1_Search.ipynb`
- `COP509_Notebook1_Search.pdf`
- `COP509_Notebook2_Extraction_Alignment.ipynb`
- `COP509_Notebook2_Extraction_Alignment.pdf`

Notebook 1 covers Task 1 search. Notebook 2 covers Task 2 recommendation extraction, response matching/alignment, classification, and evaluation. The PDFs are exported copies for marking.

## Project Layout

- `data/raw/`: source inquiry/report and government response PDFs.
- `data/ground_truth/`: search and evaluation support files.
- `src/`: shared NLP pipeline modules used by the notebooks and app.
- `backend/`: optional local FastAPI backend for the supporting web app.
- `frontend/`: optional local frontend for inspecting the pipeline.
- `outputs/`: final evidence exports.
- `screenshots/`: placeholder for local app evidence screenshots.
- `scripts/validate_recommendation_export.py`: final export validator.

The canonical final output files are:

- `outputs/final_recommendations_246.json`
- `outputs/evaluation_predictions.csv`

## Python Setup

Create and activate a local Python environment, then install the notebook dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Tesseract OCR is optional. If it is unavailable, the notebooks still run and report OCR usage in the extraction audit.

## Run The Notebooks

Launch Jupyter from the repository root:

```powershell
jupyter lab
```

Run in this order:

1. `COP509_Notebook1_Search.ipynb`
2. `COP509_Notebook2_Extraction_Alignment.ipynb`

Both notebooks detect the project root automatically and use relative paths for `data/`, `src/`, and `outputs/`.

To export fresh PDFs:

```powershell
jupyter nbconvert --to pdf COP509_Notebook1_Search.ipynb
jupyter nbconvert --to pdf COP509_Notebook2_Extraction_Alignment.ipynb
```

If local PDF export fails because TeX is unavailable, use JupyterLab's browser print/PDF flow or another environment with a working PDF toolchain.

## Validate Outputs

Validate the final Task 2 export:

```powershell
python scripts/validate_recommendation_export.py --baseline outputs/final_recommendations_246.json
```

Expected final counts:

- total rows: 246
- document-pair counts: 33, 19, 40, 10, 58, 58, 19, 9
- classification distribution: accepted 104, partial 139, rejected 1, not_addressed 2

## Optional Local Web App

The app demonstrates the same search, recommendation extraction, response matching/alignment, and classification pipeline. It is not required for marking.

Install backend dependencies if needed:

```powershell
python -m pip install -r backend/requirements.txt
```

Start the backend from the repository root:

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

Install and start the frontend in a second shell:

```powershell
cd frontend
npm install
npm run dev
```

Open the local Vite URL shown in the terminal. The frontend calls `http://localhost:8000` by default. To override it locally, set `VITE_API_BASE` as shown in `frontend/.env.example`.

The current lightweight frontend loads React, ReactDOM, Babel, and fonts from public CDNs. The backend, data, final outputs, and pipeline all run locally.

## Evidence Screenshots

Use `screenshots/` for final local app evidence, for example:

- search demo
- recommendation extraction view
- response matching/alignment view
- classification/evaluation view

No screenshot is required to run the notebooks.

## Limitations

- The Task 1 search evaluation uses a transparent 50-query automatic relevance matrix rather than a large human-labelled passage benchmark.
- A full 246-row manual ground-truth label file is unavailable, so Notebook 2 reports prediction-only validation evidence.
- Widget cells are included for interactive notebook use; static fallback tables are included so PDF exports remain readable.
- The web app is supporting evidence only and should not be treated as the marked deliverable.
