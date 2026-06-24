"""
routers/preprocessing.py
=========================
API endpoints for both preprocessing pipelines.
"""
import asyncio
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from ..schemas.preprocessing import Pipeline3Request, RunPipelineRequest, JobResponse
from ..services import config_service, job_manager

router = APIRouter(prefix="/api/preprocessing", tags=["preprocessing"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TMP_DIR = Path(__file__).resolve().parents[1] / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE3_SCRIPT = PROJECT_ROOT / "pleaides_preprocessing" / "pipeline3.py"
RUN_PIPELINE_SCRIPT = PROJECT_ROOT / "preprocessing_pipeline" / "run_pipeline.py"

# Use the satellite-sr conda env which has rasterio/gdal installed.
# Override by setting SATELLITE_SR_PYTHON env var. Falls back to sys.executable.
def _find_pipeline_python() -> str:
    override = os.environ.get("SATELLITE_SR_PYTHON", "")
    if override and Path(override).is_file():
        return override
    candidates = [
        Path.home() / ".conda" / "envs" / "satellite-sr" / "python.exe",
        Path("C:/ProgramData/anaconda3/envs/satellite-sr/python.exe"),
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    return sys.executable

PIPELINE_PYTHON: str = _find_pipeline_python()


# ── Train/Test split helper ────────────────────────────────────────────────────

def _do_train_test_split(output_dir: Path, train_ratio: float, test_output_dir: str):
    """
    After pipeline3.py completes, split patches in output_dir/hr + output_dir/lr
    into train and test sets by moving files.
    """
    hr_dir = output_dir / "hr"
    lr_dir = output_dir / "lr"

    if not hr_dir.is_dir():
        return

    all_patches = sorted([p.stem for p in hr_dir.iterdir() if p.is_file()])
    random.shuffle(all_patches)
    n_train = math.ceil(len(all_patches) * train_ratio)
    test_patches = all_patches[n_train:]

    if not test_patches:
        return

    test_dir = Path(test_output_dir) if test_output_dir else output_dir.parent / (output_dir.name + "_test")
    test_hr = test_dir / "hr"
    test_lr = test_dir / "lr"
    test_hr.mkdir(parents=True, exist_ok=True)
    test_lr.mkdir(parents=True, exist_ok=True)

    for stem in test_patches:
        # Move any extension matching the stem
        for p in hr_dir.glob(f"{stem}.*"):
            shutil.move(str(p), str(test_hr / p.name))
        for p in lr_dir.glob(f"{stem}.*"):
            shutil.move(str(p), str(test_lr / p.name))


def _do_run_pipeline_split(output_hr_dir: Path, output_lr_dir: Path, train_ratio: float):
    """Split run_pipeline outputs into train/test sub-folders."""
    all_names = sorted([p.stem for p in output_hr_dir.iterdir() if p.is_file()])
    random.shuffle(all_names)
    n_train = math.ceil(len(all_names) * train_ratio)
    test_names = set(all_names[n_train:])

    if not test_names:
        return

    test_hr = output_hr_dir.parent / (output_hr_dir.name + "_test")
    test_lr = output_lr_dir.parent / (output_lr_dir.name + "_test")
    test_hr.mkdir(parents=True, exist_ok=True)
    test_lr.mkdir(parents=True, exist_ok=True)

    for stem in test_names:
        for p in output_hr_dir.glob(f"{stem}.*"):
            shutil.move(str(p), str(test_hr / p.name))
        for p in output_lr_dir.glob(f"{stem}.*"):
            shutil.move(str(p), str(test_lr / p.name))


# ── Pipeline A — pipeline3.py ──────────────────────────────────────────────────

def _build_pipeline3_config(req: Pipeline3Request) -> dict:
    return {
        "HR_IMAGE_PATH": req.hr_image_path,
        "LR_IMAGE_PATH": req.lr_image_path,
        "OUTPUT_DIR": req.output_dir,
        "SUPPORTED_EXTENSIONS": req.supported_extensions,
        "HR_RGB_BANDS": req.hr_rgb_bands,
        "LR_RGB_BANDS": req.lr_rgb_bands,
        "SCALE_FACTOR": req.scale_factor,
        "HR_PATCH_SIZE": req.hr_patch_size,
        "STRIDE": req.stride,
        "NODATA_VALUE": req.nodata_value,
        "SATURATED_VALUE": req.saturated_value,
        "CLIP_PERCENTILES": req.clip_percentiles,
        "MAX_NODATA_FRACTION": req.max_nodata_fraction,
        "MIN_VARIANCE": req.min_variance,
        "MIN_ECC_SCORE": req.min_ecc_score,
        "MIN_SSIM": req.min_ssim,
        "COREG_A_ENABLED": req.coreg_a.enabled,
        "COREG_A_MAX_FEATURES": req.coreg_a.max_features,
        "COREG_A_MATCH_RATIO": req.coreg_a.match_ratio,
        "COREG_A_RANSAC_THRESH": req.coreg_a.ransac_thresh,
        "COREG_A_DOWNSAMPLE": req.coreg_a.downsample,
        "COREG_B_ENABLED": req.coreg_b.enabled,
        "COREG_B_DOWNSAMPLE": req.coreg_b.downsample,
        "COREG_B_UPSAMPLE_FACTOR": req.coreg_b.upsample_factor,
        "COREG_C_ENABLED": req.coreg_c.enabled,
        "COREG_C_MAX_ITER": req.coreg_c.max_iter,
        "COREG_C_EPS": req.coreg_c.eps,
        "COREG_C_WARP_MODE": req.coreg_c.warp_mode,
        "COREG_C_DISCARD_ON_FAIL": req.coreg_c.discard_on_fail,
        "RADIOMETRIC_BLOCK_SIZE": req.radiometric_block_size,
        "RADIOMETRIC_RMSE_THRESHOLD": req.radiometric_rmse_threshold,
        "RADIOMETRIC_N_SAMPLES": req.radiometric_n_samples,
        "RADIOMETRIC_POST_HIST_MATCH": req.radiometric_post_hist_match,
        "DEGRADATION_ENABLED": req.degradation_enabled,
        "DEGRADATION_TYPE": req.degradation_type,
        "bsrgan": req.bsrgan.model_dump(),
        "real_esrgan": req.real_esrgan.model_dump(),
        "bsrgan_plus": req.bsrgan_plus.model_dump(),
        "satellite": req.satellite.model_dump(),
    }


async def _run_pipeline3_with_split(job_id: str, req: Pipeline3Request, config_path: Path):
    """Launch pipeline3.py and then optionally do train/test split."""
    await job_manager.launch_job(job_id)
    job = job_manager.get_job(job_id)
    if job and job.status.value == "completed" and req.train_test_split:
        try:
            _do_train_test_split(
                Path(req.output_dir),
                req.train_ratio,
                req.test_output_dir or "",
            )
            job.logs.append(
                f"[gui] Train/test split complete (ratio={req.train_ratio})"
            )
        except Exception as e:
            job.logs.append(f"[gui] Split error: {e}")


@router.post("/pipeline3/start", response_model=JobResponse)
async def start_pipeline3(req: Pipeline3Request):
    """Write config JSON and launch pleaides_preprocessing/pipeline3.py."""
    cfg = _build_pipeline3_config(req)
    config_path = TMP_DIR / "pipeline3_config.json"
    config_service.save_json(cfg, config_path)

    cmd = [
        PIPELINE_PYTHON,
        str(PIPELINE3_SCRIPT),
        "--config",
        str(config_path),
    ]
    job_id = job_manager.create_job(cmd=cmd, cwd=str(PROJECT_ROOT), output_dir=req.output_dir)
    asyncio.create_task(_run_pipeline3_with_split(job_id, req, config_path))

    return JobResponse(job_id=job_id, status="pending")


# ── Pipeline B — run_pipeline.py ──────────────────────────────────────────────

def _build_run_pipeline_config(req: RunPipelineRequest) -> dict:
    cfg: dict = {
        "task": req.task,
        "pipeline_mode": req.pipeline_mode,
        "degradation_type": req.degradation_type,
        "scale": req.scale,
        "n_channels": req.n_channels,
        "seed": req.seed,
        "num_workers": req.num_workers,
        "input_hr_dir": req.input_hr_dir,
        "input_lr_dir": req.input_lr_dir,
        "output_hr_dir": req.output_hr_dir,
        "output_lr_dir": req.output_lr_dir,
        "supported_extensions": req.supported_extensions,
        "save_format": req.save_format,
        "save_hr_copy": req.save_hr_copy,
        "normalize_enabled": req.normalize_enabled,
        "normalize_low_percentile": req.normalize_low_percentile,
        "normalize_high_percentile": req.normalize_high_percentile,
        "cloud_mask_enabled": req.cloud_mask_enabled,
        "cloud_mask_threshold": req.cloud_mask_threshold,
        "cloud_mask_average_over": req.cloud_mask_average_over,
        "cloud_mask_dilation_size": req.cloud_mask_dilation_size,
        "cloud_mask_nodata": req.cloud_mask_nodata,
        "cloud_mask_auto_scale": req.cloud_mask_auto_scale,
        "bsrgan": req.bsrgan.model_dump(),
        "real_esrgan": req.real_esrgan.model_dump(),
        "bsrgan_plus": req.bsrgan_plus.model_dump(),
        "satellite": req.satellite.model_dump(),
    }
    return cfg


async def _run_pipeline_with_split(job_id: str, req: RunPipelineRequest, config_path: Path):
    await job_manager.launch_job(job_id)
    job = job_manager.get_job(job_id)
    if job and job.status.value == "completed" and req.train_test_split:
        try:
            _do_run_pipeline_split(
                Path(req.output_hr_dir),
                Path(req.output_lr_dir),
                req.train_ratio,
            )
            job.logs.append(f"[gui] Train/test split complete (ratio={req.train_ratio})")
        except Exception as e:
            job.logs.append(f"[gui] Split error: {e}")


@router.post("/run-pipeline/start", response_model=JobResponse)
async def start_run_pipeline(req: RunPipelineRequest):
    """Write config JSON and launch preprocessing_pipeline/run_pipeline.py."""
    cfg = _build_run_pipeline_config(req)
    config_path = TMP_DIR / f"run_pipeline_{req.task}.json"
    config_service.save_json(cfg, config_path)

    cmd = [
        PIPELINE_PYTHON,
        str(RUN_PIPELINE_SCRIPT),
        "--config",
        str(config_path),
    ]
    job_id = job_manager.create_job(cmd=cmd, cwd=str(PROJECT_ROOT))
    asyncio.create_task(_run_pipeline_with_split(job_id, req, config_path))

    return JobResponse(job_id=job_id, status="pending")


@router.get("/stream/{job_id}")
async def stream_preprocessing_logs(job_id: str):
    return StreamingResponse(
        job_manager.stream_logs(job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/preview/{job_id}/{filename}")
def get_preview_image(job_id: str, filename: str):
    """
    Serve a preview JPEG written by pipeline3.py to OUTPUT_DIR/_previews/.
    OUTPUT_DIR is user-supplied and can be anywhere on disk, so filename is
    restricted to a bare name (no path separators or traversal) and the
    resolved path is required to stay inside that job's _previews directory.
    """
    job = job_manager.get_job(job_id)
    if job is None or not job.output_dir:
        raise HTTPException(status_code=404, detail="Job not found")

    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    base = (Path(job.output_dir) / "_previews").resolve()
    candidate = (base / filename).resolve()
    if not candidate.is_relative_to(base) or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Preview not found")

    return FileResponse(candidate, media_type="image/jpeg")


@router.get("/status/{job_id}")
def get_status(job_id: str):
    summary = job_manager.get_job_summary(job_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return summary


@router.post("/stop/{job_id}")
def stop_job(job_id: str):
    if not job_manager.cancel_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": "cancelled", "job_id": job_id}
