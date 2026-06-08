"""Training API — wraps main_train_swinir.py / main_train_gan.py."""
from __future__ import annotations
import json
import os
import signal
import uuid
from pathlib import Path

import redis
from fastapi import APIRouter, HTTPException

from schemas import TrainingRequest, JobResponse, CheckpointInfo
from tasks.training_task import run_training

router = APIRouter(prefix="/training", tags=["training"])
_REDIS = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@router.post("/start", response_model=JobResponse)
async def start_training(req: TrainingRequest):
    job_id = str(uuid.uuid4())
    name = f"train · SwinIR ×{req.netG.upscale}"
    _REDIS.set("active_job", json.dumps({"id": job_id, "name": name,
               "detail": f"scale ×{req.scale} · {req.model}", "module": "train"}), ex=86400)
    _REDIS.set(f"job:{job_id}:status", "running", ex=86400)
    run_training.apply_async(args=[job_id, req.model_dump()], task_id=job_id)
    return JobResponse(job_id=job_id)


@router.post("/stop/{job_id}")
async def stop_training(job_id: str):
    pid_str = _REDIS.get(f"job:{job_id}:pid")
    if pid_str:
        try:
            os.kill(int(pid_str), signal.SIGTERM)
        except ProcessLookupError:
            pass
        _REDIS.delete(f"job:{job_id}:pid")
    _REDIS.delete("active_job")
    return {"stopped": True}


@router.get("/checkpoints", response_model=list[CheckpointInfo])
async def list_checkpoints():
    checkpoints: list[CheckpointInfo] = []
    sr_root = _PROJECT_ROOT / "superresolution"
    if not sr_root.exists():
        return []

    best_psnr = 0.0
    best_path: Path | None = None

    for models_dir in sr_root.rglob("models"):
        for pth in sorted(models_dir.glob("*_G.pth")):
            size_mb = round(pth.stat().st_size / 1e6, 1)
            # Extract iteration from filename like 100000_G.pth
            psnr = 0.0
            checkpoints.append(CheckpointInfo(
                name=pth.stem, psnr=psnr, path=str(pth.relative_to(_PROJECT_ROOT)),
                size_mb=size_mb, is_best=False,
            ))
            if psnr >= best_psnr:
                best_psnr = psnr
                best_path = pth

    if best_path and checkpoints:
        for c in checkpoints:
            if c.path == str(best_path.relative_to(_PROJECT_ROOT)):
                c.is_best = True

    return sorted(checkpoints, key=lambda c: c.name, reverse=True)[:20]


@router.delete("/checkpoints/{name}")
async def delete_checkpoint(name: str):
    sr_root = _PROJECT_ROOT / "superresolution"
    matches = list(sr_root.rglob(f"{name}.pth"))
    if not matches:
        raise HTTPException(status_code=404, detail="Checkpoint not found")
    matches[0].unlink()
    return {"deleted": name}
