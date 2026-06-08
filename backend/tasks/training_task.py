"""Celery task for SwinIR training — wraps main_train_swinir.py / main_train_gan.py."""
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
    _REDIS.expire(f"logs:{job_id}", 86400)


@celery_app.task(bind=True, name="tasks.run_training")
def run_training(self, job_id: str, config: dict) -> dict:
    model_type = config.get("model", "plain")
    task_name = config.get("task", "unnamed")

    # Pick the right training script
    if "swinir" in task_name.lower() or config.get("netG", {}).get("net_type") == "swinir":
        script = "main_train_swinir_gan.py" if model_type == "gan" else "main_train_swinir.py"
    elif model_type == "gan":
        script = "main_train_gan.py"
    else:
        script = "main_train_psnr.py"

    _push_log(job_id, f"▸ Training task: {task_name}", "step")
    _push_log(job_id, f"  script: {script} · model: {model_type}", "info")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", dir=str(_PROJECT_ROOT / "options"),
                                     delete=False, prefix="ui_train_") as f:
        json.dump(config, f, indent=2)
        opt_path = f.name

    try:
        proc = subprocess.Popen(
            [sys.executable, str(_PROJECT_ROOT / script), "--opt", opt_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(_PROJECT_ROOT),
        )
        assert proc.stdout
        _REDIS.set(f"job:{job_id}:pid", str(proc.pid), ex=86400)

        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            lv = "ok" if ("best" in line.lower() or "saved" in line.lower()) else \
                 "warn" if "warn" in line.lower() else \
                 "step" if "epoch" in line.lower() or "iter" in line.lower() else "info"
            _push_log(job_id, line, lv)

        proc.wait()
        rc = proc.returncode
        if rc == 0:
            _push_log(job_id, "✓ Training completed", "ok")
            return {"status": "done"}
        else:
            _push_log(job_id, f"Training exited with code {rc}", "warn")
            return {"status": "failed", "exit_code": rc}
    finally:
        os.unlink(opt_path)
        _REDIS.delete(f"job:{job_id}:pid")
    _REDIS.set(f"job:{job_id}:done", "1", ex=86400)
    _REDIS.rpush(f"logs:{job_id}", "[DONE]")
