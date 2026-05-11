"""FastAPI entry point for the Policy Response Analyser backend."""
import asyncio
import os
import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.state import pipeline
from backend.core.logger import log as app_log
from backend.routes import search, pipeline as pipeline_route, log, preview


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Preload the full 8-pair search chunk corpus before the server begins
    accepting requests so the Search tab is immediately usable and no
    "corpus empty; loading default search corpus" warmup happens on the
    first user query.

    Only the load + chunk stages run here — extraction / alignment /
    classification are deliberately skipped because Task 2 results come from
    the static ``final_recommendations_246.json`` shipped with the frontend.
    Embeddings stay lazy (built on first Hybrid/Semantic query per preset)
    so cold start stays bounded.
    """
    print("[STARTUP] preloading search corpus...", file=sys.stderr, flush=True)
    app_log.emit("startup.preload", "preloading search corpus", "info")
    t0 = time.monotonic()
    try:
        pairs_loaded, chunks_loaded = await asyncio.to_thread(
            pipeline.ensure_default_search_corpus
        )
        ms = int((time.monotonic() - t0) * 1000)
        if pairs_loaded == 0:
            total_chunks = sum(
                len(s.policy_chunks) + len(s.response_chunks)
                for s in pipeline.preset_cache.values()
            )
            print(
                f"[STARTUP] search corpus already cached: "
                f"{len(pipeline.preset_cache)} pairs, {total_chunks} chunks "
                f"({ms} ms)",
                file=sys.stderr,
                flush=True,
            )
            app_log.emit(
                "startup.preload",
                f"already cached: {len(pipeline.preset_cache)} pairs, "
                f"{total_chunks} chunks",
                "ok",
                ms=ms,
            )
        else:
            print(
                f"[STARTUP] search corpus ready: {pairs_loaded} pairs, "
                f"{chunks_loaded} chunks ({ms} ms)",
                file=sys.stderr,
                flush=True,
            )
            app_log.emit(
                "startup.preload",
                f"search corpus ready: {pairs_loaded} pairs, "
                f"{chunks_loaded} chunks",
                "ok",
                ms=ms,
            )
    except Exception as exc:
        # Don't crash the server if a single PDF is missing on a dev box —
        # the search-route safety net will surface a clear 500 to the user
        # if they actually try to search before the corpus is ready.
        print(
            f"[STARTUP] search corpus preload FAILED ({type(exc).__name__}: "
            f"{exc}); will retry lazily on first search",
            file=sys.stderr,
            flush=True,
        )
        app_log.emit(
            "startup.preload",
            f"failed ({type(exc).__name__}: {exc}); falling back to lazy load",
            "err",
        )
    yield


app = FastAPI(title="Policy Response Analyser API", lifespan=lifespan)

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

app.include_router(pipeline_route.router, prefix="/api/pipeline", tags=["pipeline"])
app.include_router(search.router,         prefix="/api/search",   tags=["search"])
app.include_router(preview.router,        prefix="/api/preview",  tags=["preview"])
app.include_router(log.router,            prefix="/api/log",      tags=["log"])


@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok"}
