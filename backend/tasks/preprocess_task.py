"""Celery tasks for preprocessing pipeline — wraps existing run_pipeline.py and complete_pipeline.py."""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import redis

from tasks.celery_app import celery_app

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REDIS = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)


def _push_log(job_id: str, text: str, lv: str = "info") -> None:
    ts = time.strftime("%H:%M:%S")
    line = json.dumps({"ts": ts, "text": text, "lv": lv})
    _REDIS.rpush(f"logs:{job_id}", line)
    _REDIS.expire(f"logs:{job_id}", 3600)


def _run_script(job_id: str, cmd: list[str]) -> int:
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, cwd=str(_PROJECT_ROOT),
    )
    assert proc.stdout
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        lv = "ok" if "done" in line.lower() or "✓" in line else \
             "warn" if "[warn]" in line.lower() else \
             "step" if line.startswith("  ") is False and line.startswith("[") else "info"
        _push_log(job_id, line, lv)
    proc.wait()
    return proc.returncode


@celery_app.task(bind=True, name="tasks.run_simple_pipeline")
def run_simple_pipeline(self, job_id: str, config: dict) -> dict:
    _push_log(job_id, f"▸ Starting simple pipeline: {config.get('task', 'unnamed')}", "step")
    _push_log(job_id, f"  mode: {config.get('pipeline_mode')} · degradation: {config.get('degradation_type')} · scale: ×{config.get('scale')}", "info")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f, indent=2)
        cfg_path = f.name

    try:
        rc = _run_script(job_id, [
            sys.executable,
            str(_PROJECT_ROOT / "preprocessing_pipeline" / "run_pipeline.py"),
            "--config", cfg_path,
        ])
        if rc == 0:
            _push_log(job_id, "✓ Pipeline completed successfully", "ok")
            return {"status": "done", "output_hr_dir": config.get("output_hr_dir"), "output_lr_dir": config.get("output_lr_dir")}
        else:
            _push_log(job_id, f"Pipeline exited with code {rc}", "warn")
            return {"status": "failed", "exit_code": rc}
    finally:
        os.unlink(cfg_path)
    _REDIS.set(f"job:{job_id}:done", "1", ex=3600)
    _REDIS.rpush(f"logs:{job_id}", "[DONE]")


@celery_app.task(bind=True, name="tasks.run_complete_pipeline")
def run_complete_pipeline(self, job_id: str, config: dict) -> dict:
    _push_log(job_id, f"▸ Starting complete satellite pipeline: {config.get('task', 'unnamed')}", "step")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f, indent=2)
        cfg_path = f.name

    try:
        rc = _run_script(job_id, [
            sys.executable,
            str(_PROJECT_ROOT / "preprocessing_pipeline" / "complete_pipeline.py"),
            "--config", cfg_path,
        ])
        if rc == 0:
            _push_log(job_id, "✓ Complete pipeline finished", "ok")
            return {"status": "done"}
        else:
            _push_log(job_id, f"Pipeline exited with code {rc}", "warn")
            return {"status": "failed", "exit_code": rc}
    finally:
        os.unlink(cfg_path)
    _REDIS.set(f"job:{job_id}:done", "1", ex=3600)
    _REDIS.rpush(f"logs:{job_id}", "[DONE]")
