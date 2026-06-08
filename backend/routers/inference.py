"""Inference API — wraps main_test_swinir_config.py."""
from __future__ import annotations
import json
import shutil
import uuid
from pathlib import Path

import aiofiles
import redis
from fastapi import APIRouter, HTTPException, UploadFile, File

from schemas import InferenceRequest, JobResponse
from tasks.inference_task import run_inference as run_inference_task

router = APIRouter(prefix="/inference", tags=["inference"])
_REDIS = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_UPLOAD_DIR = _PROJECT_ROOT / "uploads" / "inference"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    suffix = Path(file.filename or "img.png").suffix
    dest = _UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"
    async with aiofiles.open(dest, "wb") as f:
        content = await file.read()
        await f.write(content)
    return {"path": str(dest.relative_to(_PROJECT_ROOT))}


@router.post("/run", response_model=JobResponse)
async def run_inference(req: InferenceRequest):
    job_id = str(uuid.uuid4())
    _REDIS.set("active_job", json.dumps({
        "id": job_id, "name": "inference · SwinIR",
        "detail": "reconstructing", "module": "inference",
    }), ex=3600)
    _REDIS.set(f"job:{job_id}:status", "running", ex=3600)
    run_inference_task.apply_async(args=[job_id, req.model_dump()], task_id=job_id)
    return JobResponse(job_id=job_id)


@router.get("/result/{job_id}")
async def get_result(job_id: str):
    raw = _REDIS.get(f"job:{job_id}:result")
    if not raw:
        return None
    return json.loads(raw)
