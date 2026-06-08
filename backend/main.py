"""SUPARCO Super-Resolution Lab — FastAPI backend."""
from __future__ import annotations
import asyncio
import json
from typing import AsyncGenerator

import redis
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from routers import preprocessing, training, inference, status

app = FastAPI(title="SUPARCO SR Lab API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(preprocessing.router)
app.include_router(training.router)
app.include_router(inference.router)
app.include_router(status.router)

_REDIS = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    status_val = _REDIS.get(f"job:{job_id}:status") or "pending"
    return {"id": job_id, "status": status_val}


@app.get("/jobs/{job_id}/logs")
async def stream_logs(job_id: str):
    """SSE endpoint: streams log lines for a job, sends [DONE] when finished."""

    async def generator() -> AsyncGenerator[dict, None]:
        cursor = 0
        while True:
            lines = _REDIS.lrange(f"logs:{job_id}", cursor, -1)
            for line in lines:
                if line == "[DONE]":
                    yield {"data": "[DONE]"}
                    return
                yield {"data": line}
            cursor += len(lines)

            done = _REDIS.get(f"job:{job_id}:done")
            if done:
                remaining = _REDIS.lrange(f"logs:{job_id}", cursor, -1)
                for line in remaining:
                    yield {"data": line}
                yield {"data": "[DONE]"}
                return

            await asyncio.sleep(0.4)

    return EventSourceResponse(generator())


@app.get("/health")
async def health():
    return {"status": "ok"}
