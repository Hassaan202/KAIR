"""
routers/inference.py
====================
API endpoints for SwinIR inference.
"""
import asyncio
import json
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, FileResponse

from ..schemas.inference import (
    StartInferenceRequest, JobResponse,
    RawPairedInferenceRequest, LROnlyInferenceRequest,
)
from ..services import config_service, job_manager

router = APIRouter(prefix="/api/inference", tags=["inference"])

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TMP_DIR = Path(__file__).resolve().parents[1] / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Script that accepts CONFIG injected as env vars — we write a temp config file
INFERENCE_SCRIPT = PROJECT_ROOT / "main_test_swinir_config.py"
INFERENCE_WRAPPER = TMP_DIR / "run_inference.py"
RAW_INFERENCE_SCRIPT = PROJECT_ROOT / "raw_inference.py"


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


def _build_raw_config(
    req,
    mode: str,
) -> tuple[dict, Path]:
    """
    Build the raw_inference.py JSON config from a request body and write it
    to a temp file.  Returns (config_dict, config_path).
    """
    mc = req.model_network_config
    model_cfg = {
        "upscale":          mc.upscale,
        "in_chans":         mc.in_chans,
        "img_size":         mc.img_size,
        "window_size":      mc.window_size,
        "img_range":        mc.img_range,
        "depths":           mc.depths,
        "embed_dim":        mc.embed_dim,
        "num_heads":        mc.num_heads,
        "mlp_ratio":        mc.mlp_ratio,
        "upsampler":        mc.upsampler,
        "resi_connection":  mc.resi_connection,
    }

    cfg: dict = {
        "mode":        mode,
        "model_path":  req.model_path,
        "output_dir":  req.output_dir,
        "patch_size":  req.patch_size,
        "overlap":     req.overlap,
        "scale_factor": req.scale_factor,
        "model_config": model_cfg,
    }

    if mode == "paired":
        cfg["lr_path"]  = req.lr_path
        cfg["hr_path"]  = req.hr_path
        cfg["lr_bands"] = req.lr_bands
        cfg["hr_bands"] = req.hr_bands
        # Merge CoregConfig fields into top-level cfg
        coreg = req.coreg
        cfg.update({
            "enable_preprocessing":        coreg.enable_preprocessing,
            "coreg_a_enabled":             coreg.coreg_a_enabled,
            "coreg_a_max_features":        coreg.coreg_a_max_features,
            "coreg_a_match_ratio":         coreg.coreg_a_match_ratio,
            "coreg_a_ransac_thresh":       coreg.coreg_a_ransac_thresh,
            "coreg_b_enabled":             coreg.coreg_b_enabled,
            "coreg_b_upsample_factor":     coreg.coreg_b_upsample_factor,
            "radiometric_enabled":         coreg.radiometric_enabled,
            "radiometric_block_size":      coreg.radiometric_block_size,
            "radiometric_rmse_threshold":  coreg.radiometric_rmse_threshold,
            "radiometric_n_samples":       coreg.radiometric_n_samples,
            "radiometric_n_fit_windows":   coreg.radiometric_n_fit_windows,
            "radiometric_post_hist_match": coreg.radiometric_post_hist_match,
            "histogram_n_sample_windows":  coreg.histogram_n_sample_windows,
            "nodata_value":                coreg.nodata_value,
            "saturated_value":             coreg.saturated_value,
            "clip_percentiles":            coreg.clip_percentiles,
            "percentile_n_sample_windows": coreg.percentile_n_sample_windows,
            "coreg_preview_decim_dim":     coreg.coreg_preview_decim_dim,
        })
    else:  # lr_only
        cfg["lr_path"]  = req.lr_path
        cfg["lr_bands"] = req.lr_bands

    # Write to tmp
    import hashlib
    cfg_hash  = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    cfg_path  = TMP_DIR / f"raw_inference_{mode}_{cfg_hash}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg, cfg_path


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
    """Launch the patch-based inference subprocess."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Raw satellite image inference endpoints
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/raw-paired/start", response_model=JobResponse)
async def start_raw_paired_inference(req: RawPairedInferenceRequest):
    """
    Launch a paired (LR + HR) raw image inference job.
    Optionally runs pipeline3-style coregistration + radiometric normalisation
    before patch extraction, then stitches SR output and computes metrics.
    """
    if not RAW_INFERENCE_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="raw_inference.py not found in project root.")

    _, cfg_path = _build_raw_config(req, mode="paired")
    cmd    = [sys.executable, str(RAW_INFERENCE_SCRIPT), "--config", str(cfg_path)]
    job_id = job_manager.create_job(
        cmd=cmd,
        cwd=str(PROJECT_ROOT),
        meta={"output_dir": req.output_dir, "mode": "paired"},
    )
    asyncio.create_task(job_manager.launch_job(job_id))
    return JobResponse(job_id=job_id, status="pending")


@router.post("/lr-only/start", response_model=JobResponse)
async def start_lr_only_inference(req: LROnlyInferenceRequest):
    """
    Launch an LR-only raw image inference job.
    No HR ground truth — SR is computed and saved; no metrics are produced.
    """
    if not RAW_INFERENCE_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="raw_inference.py not found in project root.")

    _, cfg_path = _build_raw_config(req, mode="lr_only")
    cmd    = [sys.executable, str(RAW_INFERENCE_SCRIPT), "--config", str(cfg_path)]
    job_id = job_manager.create_job(
        cmd=cmd,
        cwd=str(PROJECT_ROOT),
        meta={"output_dir": req.output_dir, "mode": "lr_only"},
    )
    asyncio.create_task(job_manager.launch_job(job_id))
    return JobResponse(job_id=job_id, status="pending")


@router.get("/raw/result/{job_id}/{filename}")
def get_raw_result_image(job_id: str, filename: str):
    """
    Serve a result PNG (lr_display.png, sr_display.png, hr_display.png)
    from the job's output_dir.
    """
    # Validate filename to prevent path traversal
    allowed = {"lr_display.png", "sr_display.png", "hr_display.png"}
    if filename not in allowed:
        raise HTTPException(status_code=400, detail=f"Invalid filename. Allowed: {sorted(allowed)}")

    summary = job_manager.get_job_summary(job_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = (summary.get("meta") or {}).get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=404, detail="Job has no output_dir recorded.")

    file_path = PROJECT_ROOT / output_dir / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not yet available.")

    return FileResponse(str(file_path), media_type="image/png")


@router.get("/raw/metrics/{job_id}")
def get_raw_metrics(job_id: str):
    """
    Return the metrics.json produced by a paired raw inference job.
    """
    summary = job_manager.get_job_summary(job_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Job not found")

    output_dir = (summary.get("meta") or {}).get("output_dir")
    if not output_dir:
        raise HTTPException(status_code=404, detail="Job has no output_dir recorded.")

    metrics_path = PROJECT_ROOT / output_dir / "metrics.json"
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="metrics.json not yet available.")

    with open(metrics_path, "r", encoding="utf-8") as fh:
        return json.load(fh)



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
