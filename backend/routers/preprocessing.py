"""Preprocessing API — wraps run_pipeline.py and complete_pipeline.py."""
from __future__ import annotations
import json
import uuid

import redis
from fastapi import APIRouter

from schemas import SimplePreprocessRequest, CompletePipelineRequest, JobResponse
from tasks.preprocess_task import run_simple_pipeline, run_complete_pipeline

router = APIRouter(prefix="/preprocessing", tags=["preprocessing"])
_REDIS = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


def _register_job(job_id: str, name: str, detail: str) -> None:
    _REDIS.set("active_job", json.dumps({"id": job_id, "name": name, "detail": detail, "module": "preprocess"}), ex=86400)
    _REDIS.set(f"job:{job_id}:status", "running", ex=86400)


@router.post("/run/simple", response_model=JobResponse)
async def run_simple(req: SimplePreprocessRequest):
    job_id = str(uuid.uuid4())
    _register_job(job_id, f"preprocess · {req.task}", f"mode: {req.pipeline_mode}")
    run_simple_pipeline.apply_async(args=[job_id, req.model_dump()], task_id=job_id)
    return JobResponse(job_id=job_id)


@router.post("/run/complete", response_model=JobResponse)
async def run_complete(req: CompletePipelineRequest):
    job_id = str(uuid.uuid4())
    _register_job(job_id, f"preprocess · {req.task}", "6-stage satellite pipeline")
    run_complete_pipeline.apply_async(args=[job_id, req.model_dump()], task_id=job_id)
    return JobResponse(job_id=job_id)
