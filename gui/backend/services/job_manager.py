"""
job_manager.py
==============
Manages long-running subprocess jobs (training, inference, preprocessing).
Streams stdout/stderr as Server-Sent Events.
"""
import asyncio
import os
import signal
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Deque, Dict, Optional


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    job_id: str
    cmd: list
    cwd: str
    status: JobStatus = JobStatus.PENDING
    process: Optional[asyncio.subprocess.Process] = None
    logs: Deque[str] = field(default_factory=lambda: deque(maxlen=5000))
    return_code: Optional[int] = None


# Global in-memory job store
_jobs: Dict[str, Job] = {}


def create_job(cmd: list, cwd: str) -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = Job(job_id=job_id, cmd=cmd, cwd=cwd)
    return job_id


def get_job(job_id: str) -> Optional[Job]:
    return _jobs.get(job_id)


def list_jobs() -> list:
    return [
        {
            "job_id": j.job_id,
            "status": j.status,
            "return_code": j.return_code,
            "log_lines": len(j.logs),
        }
        for j in _jobs.values()
    ]


async def launch_job(job_id: str) -> None:
    """Launch the subprocess for a given job and stream its output into job.logs."""
    job = _jobs.get(job_id)
    if not job:
        raise ValueError(f"Job {job_id} not found")

    job.status = JobStatus.RUNNING
    env = {**os.environ}

    try:
        proc = await asyncio.create_subprocess_exec(
            *job.cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=job.cwd,
            env=env,
        )
        job.process = proc

        # Stream lines into the deque
        assert proc.stdout is not None
        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            job.logs.append(line)

        await proc.wait()
        job.return_code = proc.returncode
        if proc.returncode == 0:
            job.status = JobStatus.COMPLETED
        else:
            job.status = JobStatus.FAILED

    except asyncio.CancelledError:
        job.status = JobStatus.CANCELLED
        if job.process and job.process.returncode is None:
            job.process.terminate()
        raise
    except Exception as exc:
        job.logs.append(f"[job_manager] Error launching job: {exc}")
        job.status = JobStatus.FAILED


async def stream_logs(job_id: str) -> AsyncGenerator[str, None]:
    """
    Async generator for SSE streaming.
    Yields new log lines as they arrive, then sends a final status event.
    """
    job = _jobs.get(job_id)
    if not job:
        yield f"data: [ERROR] Job {job_id} not found\n\n"
        return

    sent = 0
    while True:
        logs_snapshot = list(job.logs)
        new_lines = logs_snapshot[sent:]
        for line in new_lines:
            yield f"data: {line}\n\n"
        sent += len(new_lines)

        if job.status in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
        ):
            # Flush any remaining lines
            logs_snapshot = list(job.logs)
            for line in logs_snapshot[sent:]:
                yield f"data: {line}\n\n"
            yield f"event: status\ndata: {job.status}\n\n"
            break

        await asyncio.sleep(0.3)


def cancel_job(job_id: str) -> bool:
    job = _jobs.get(job_id)
    if not job:
        return False
    if job.process and job.process.returncode is None:
        try:
            os.kill(job.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    job.status = JobStatus.CANCELLED
    return True


def get_job_summary(job_id: str) -> Optional[dict]:
    job = _jobs.get(job_id)
    if not job:
        return None
    return {
        "job_id": job.job_id,
        "status": job.status,
        "return_code": job.return_code,
        "logs": list(job.logs)[-200:],  # last 200 lines for polling
    }
