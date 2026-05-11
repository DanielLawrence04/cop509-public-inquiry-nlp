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

1. `COP509_Notebook1_Search.ipynb`
2. `COP509_Notebook2_Extraction_Alignment.ipynb`

Both notebooks detect the project root automatically and use relative paths for `data/`, `src/`, and `outputs/`. Inside this submission folder the two notebooks sit at the top level alongside `data/`, `src/`, `outputs/`, `scripts/`, and `backend/`; the `notebooks/` subdirectory used in the GitHub repository is collapsed away here.

## Export PDFs

From the submission folder root:

```powershell
jupyter nbconvert --to pdf COP509_Notebook1_Search.ipynb
jupyter nbconvert --to pdf COP509_Notebook2_Extraction_Alignment.ipynb
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

The optional web app lives in `frontend/` and `backend/` in the public GitHub repository (link below). The submission ZIP only carries the slice of the backend needed by the notebooks (`backend/core/`), so the local-run / deployment instructions below refer to the GitHub repository, not this folder. The notebooks and PDF exports remain the official coursework submission; the web app is an optional extension and is not required for marking.

### Run the web app locally

```powershell
# backend (from repo root)
python -m uvicorn backend.main:app --reload --port 8000

# frontend (in a second shell)
cd frontend
npm install
npm run dev
```

The frontend reads `VITE_API_BASE` at build time and falls back to
`http://localhost:8000` when unset, so no extra configuration is needed for
local development. See `frontend/.env.example` for the variable name.

### Full production deployment

The frontend is a static Vite build, and the backend is a FastAPI service.
For the app to work end-to-end on Vercel, the backend must be hosted
separately and the frontend must be built with `VITE_API_BASE` pointing at it.

**1. Deploy the backend (Render example, see `render.yaml`):**

- Root directory: `.`
- Build command: `pip install -r backend/requirements.txt`
- Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- Health check path: `/health`
- Environment variables:
  - `CORS_ORIGINS` = the Vercel frontend URL (e.g. `https://your-app.vercel.app`).
    Comma-separated values are supported; local dev origins are always allowed.

Railway and Azure App Service work the same way — same build/start commands,
same `CORS_ORIGINS` variable. Note the start command for the chosen host and
the resulting public backend URL.

**2. Deploy the frontend to Vercel:**

- Root directory: `frontend`
- Framework preset: Vite
- Build command: `npm run build`
- Output directory: `dist`
- Environment variable:
  - `VITE_API_BASE` = the deployed backend URL, no trailing slash
    (e.g. `https://cop509-backend.onrender.com`).

**3. Wire the two together:**

After the first deploy, set `CORS_ORIGINS` on the backend to the exact Vercel
URL, then redeploy the backend. The browser `OPTIONS` preflight from Vercel to
the backend will then succeed and the app is fully live.

## Optional Supporting Links

The notebooks and PDF exports are the official coursework submission files. The GitHub repository and deployed web app below are included as supporting evidence for the extended implementation.

- GitHub repository: https://github.com/DanielLawrence04/cop509-public-inquiry-nlp/tree/main
- Web app demo: https://cop509-public-inquiry-nlp.vercel.app/

The hosted demo opens with the validated 246-row coursework export already loaded (served as a static asset by the Vercel frontend), so the Documents / Recommendation Analysis / Evaluation Summary tabs show the validated evidence immediately. The Render backend powers live recomputation (search, batch evaluate) if used; the first request after a period of inactivity may take 30–60 seconds while the free dyno wakes up. Subsequent requests are fast.

## Repository Hygiene

The `.gitignore` excludes secrets, virtual environments, Python/Jupyter caches, logs, local archives, `frontend/node_modules/`, `frontend/dist/`, runtime output caches, and old timestamped exports.

It keeps the notebooks, source modules, required data, README, requirements, and canonical final output evidence.

## Limitations

- The search evaluation uses a transparent 50-query automatic relevance matrix rather than a large human-labelled passage benchmark.
- Full 246-row manual ground truth is unavailable, so Task 2 reports prediction-only validation evidence.
- Widget cells are included for interactive use; static fallback tables are included so PDF exports remain readable.
- The web app is optional supporting evidence and should not be treated as the official marked deliverable.
