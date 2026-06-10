"""
routers/inference.py
====================
API endpoints for SwinIR inference.
"""
import asyncio
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..schemas.inference import StartInferenceRequest, JobResponse
from ..services import config_service, job_manager

router = APIRouter(prefix="/api/inference", tags=["inference"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TMP_DIR = Path(__file__).resolve().parents[1] / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Script that accepts CONFIG injected as env vars — we write a temp config file
INFERENCE_SCRIPT = PROJECT_ROOT / "main_test_swinir_config.py"
INFERENCE_WRAPPER = TMP_DIR / "run_inference.py"


def _write_inference_script(req: StartInferenceRequest) -> Path:
    """
    Write a self-contained inference launcher that overrides CONFIG and MODEL_CONFIG
    and calls main_test_swinir_config.main().
    """
    mc = req.model_network_config
    script = f"""
import sys
sys.path.insert(0, r'{PROJECT_ROOT}')
import main_test_swinir_config as m

m.CONFIG = {{
    "model_path": r'{req.model_path}',
    "lr_dir": r'{req.lr_dir}',
    "hr_dir": r'{req.hr_dir}',
    "sr_dir": r'{req.sr_dir}',
    "tile": {repr(req.tile)},
    "tile_overlap": {req.tile_overlap},
    "overwrite_sr": {req.overwrite_sr},
    "log_dir": r'{req.log_dir}',
}}
m.MODEL_CONFIG = {{
    "upscale": {mc.upscale},
    "in_chans": {mc.in_chans},
    "img_size": {mc.img_size},
    "window_size": {mc.window_size},
    "img_range": {mc.img_range},
    "depths": {mc.depths},
    "embed_dim": {mc.embed_dim},
    "num_heads": {mc.num_heads},
    "mlp_ratio": {mc.mlp_ratio},
    "upsampler": '{mc.upsampler}',
    "resi_connection": '{mc.resi_connection}',
}}
m.main()
"""
    wrapper = TMP_DIR / f"run_inference_{hash(req.model_path) & 0xFFFFFF}.py"
    wrapper.write_text(script.strip(), encoding="utf-8")
    return wrapper


@router.get("/tasks")
def list_tasks():
    """List all trained task names from superresolution/."""
    return config_service.list_training_runs()


@router.get("/latest-model/{task_name}")
def get_latest_model(task_name: str):
    """Return the latest model path + autofilled MODEL_CONFIG for a task."""
    try:
        return config_service.get_latest_model_info(task_name)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/start", response_model=JobResponse)
async def start_inference(req: StartInferenceRequest):
    """Launch the inference subprocess."""
    wrapper = _write_inference_script(req)
    cmd = [sys.executable, str(wrapper)]
    job_id = job_manager.create_job(cmd=cmd, cwd=str(PROJECT_ROOT))
    asyncio.create_task(job_manager.launch_job(job_id))
    return JobResponse(job_id=job_id, status="pending")


@router.get("/stream/{job_id}")
async def stream_inference_logs(job_id: str):
    return StreamingResponse(
        job_manager.stream_logs(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/status/{job_id}")
def get_status(job_id: str):
    summary = job_manager.get_job_summary(job_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return summary


@router.post("/stop/{job_id}")
def stop_inference(job_id: str):
    if not job_manager.cancel_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "cancelled", "job_id": job_id}
