"""Pipeline log emitter — write events here, SSE endpoint drains the queue."""
from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class LogEvent:
    fn: str
    info: str
    status: str   # "ok" | "err" | "info"
    ms: int | None = None
    ts: str = field(default_factory=lambda: time.strftime("%H:%M:%S") + f".{int(time.time()*1000)%1000:03d}")

    def to_sse(self) -> str:
        ms_part = f" {self.ms} ms" if self.ms is not None else ""
        return f"data: {self.ts}|{self.fn}|{self.info}|{self.status}|{ms_part}\n\n"


class PipelineLogger:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[LogEvent] = asyncio.Queue()

    def emit(self, fn: str, info: str, status: str = "info", ms: int | None = None) -> None:
        """Called from pipeline code (sync or async) to push a log event."""
        event = LogEvent(fn=fn, info=info, status=status, ms=ms)
        try:
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            pass

    async def stream(self):
        """Async generator consumed by the SSE route."""
        while True:
            event = await self._queue.get()
            yield event.to_sse()


# Module-level singleton shared across all routes
log = PipelineLogger()
