"""GPU, memory, and active job status."""
from __future__ import annotations
import json
from fastapi import APIRouter
import psutil
import redis

from schemas import GpuStatusResponse

router = APIRouter(prefix="/status", tags=["status"])
_REDIS = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
    _handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _NVML = False
    _handle = None


def _get_gpu() -> tuple[float, float, float]:
    """Returns (util_pct, vram_used_gb, vram_total_gb)."""
    if _NVML and _handle:
        try:
            import pynvml
            util = pynvml.nvmlDeviceGetUtilizationRates(_handle).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(_handle)
            return float(util), mem.used / 1e9, mem.total / 1e9
        except Exception:
            pass
    # CPU fallback
    cpu = psutil.cpu_percent(interval=None)
    vm = psutil.virtual_memory()
    return cpu, vm.used / 1e9, vm.total / 1e9


@router.get("/gpu", response_model=GpuStatusResponse)
async def gpu_status():
    util, used, total = _get_gpu()
    active_raw = _REDIS.get("active_job")
    active = json.loads(active_raw) if active_raw else None
    return GpuStatusResponse(gpu_util=util, vram_used_gb=used, vram_total_gb=total, active_job=active)


@router.get("/job")
async def active_job():
    raw = _REDIS.get("active_job")
    return json.loads(raw) if raw else None
