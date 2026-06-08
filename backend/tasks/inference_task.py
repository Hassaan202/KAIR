"""Celery task for SwinIR inference — wraps main_test_swinir_config.py."""
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
    _REDIS.rpush(f"logs:{job_id}", json.dumps({"ts": ts, "text": text, "lv": lv}))
    _REDIS.expire(f"logs:{job_id}", 3600)


@celery_app.task(bind=True, name="tasks.run_inference")
def run_inference(self, job_id: str, config: dict) -> dict:
    _push_log(job_id, "▸ Starting SwinIR inference", "step")
    _push_log(job_id, f"  model: {config.get('model_path')}", "info")
    _push_log(job_id, f"  lr_dir: {config.get('lr_dir')}", "info")

    # Write a temporary config script that patches CONFIG and MODEL_CONFIG in the test script
    script_content = f"""
import sys
sys.path.insert(0, r"{_PROJECT_ROOT}")
import main_test_swinir_config as _m

_m.CONFIG.update({json.dumps(config)})
_m.MODEL_CONFIG.update({json.dumps(config.get("model_config", {}))})

# Re-run main with patched config
_m.main()
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_content)
        script_path = f.name

    try:
        proc = subprocess.Popen(
            [sys.executable, script_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(_PROJECT_ROOT),
        )
        assert proc.stdout
        metrics_line: str | None = None

        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            lv = "ok" if "average" in line.lower() else \
                 "warn" if "warn" in line.lower() or "error" in line.lower() else "info"
            _push_log(job_id, line, lv)

            if "Average PSNR" in line:
                metrics_line = line

        proc.wait()
        rc = proc.returncode

        if rc == 0:
            _push_log(job_id, "✓ Inference completed", "ok")
            result: dict = {"status": "done", "sr_dir": config.get("sr_dir")}
            _REDIS.set(f"job:{job_id}:result", json.dumps(result), ex=3600)
            return result
        else:
            return {"status": "failed", "exit_code": rc}
    finally:
        os.unlink(script_path)
    _REDIS.set(f"job:{job_id}:done", "1", ex=3600)
    _REDIS.rpush(f"logs:{job_id}", "[DONE]")
