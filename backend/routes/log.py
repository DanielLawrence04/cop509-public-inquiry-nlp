"""Server-Sent Events stream for the live pipeline log drawer."""
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from backend.core.logger import log

router = APIRouter()


@router.get("/stream")
async def log_stream():
    return StreamingResponse(
        log.stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Access-Control-Allow-Origin": "*",
        },
    )
