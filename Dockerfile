# Dockerfile for the COP509 Policy Response Analyser FastAPI backend.
# Built on Render so Tesseract OCR is installed at the system level — required
# by pytesseract when scanned PDFs need OCR. Plain Render Python runtimes mount
# /var/lib/apt as read-only, which is why apt-get install fails there.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages: Tesseract for OCR fallback in src/pdf_loader.py, plus the
# minimum build/runtime libraries PyMuPDF and Pillow need.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker layer caching works between code edits.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install -r /app/backend/requirements.txt

# Copy the rest of the repo (src/, backend/, data/, outputs/, scripts/, etc.).
# .dockerignore keeps notebooks, node_modules, dist, caches, and logs out.
COPY . /app

EXPOSE 8000

# Render injects $PORT at runtime; fall back to 8000 for local `docker run`.
CMD ["sh", "-c", "python -m uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
