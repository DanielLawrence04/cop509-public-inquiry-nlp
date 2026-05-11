"""FastAPI entry point for the Policy Response Analyser backend."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import search, pipeline, log, preview

app = FastAPI(title="Policy Response Analyser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(search.router,   prefix="/api/search",   tags=["search"])
app.include_router(preview.router,  prefix="/api/preview",  tags=["preview"])
app.include_router(log.router,      prefix="/api/log",      tags=["log"])


@app.get("/api/health")
def health():
    return {"status": "ok"}
