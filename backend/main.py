"""FastAPI entry point for the Policy Response Analyser backend."""
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import search, pipeline, log, preview

app = FastAPI(title="Policy Response Analyser API")

# CORS origins are configurable via the CORS_ORIGINS env var
# (comma-separated). Local Vite/dev origins are always included so local
# development keeps working without extra configuration.
_DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]
_extra = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
_origins = list(dict.fromkeys(_DEFAULT_ORIGINS + _extra))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(search.router,   prefix="/api/search",   tags=["search"])
app.include_router(preview.router,  prefix="/api/preview",  tags=["preview"])
app.include_router(log.router,      prefix="/api/log",      tags=["log"])


@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok"}
