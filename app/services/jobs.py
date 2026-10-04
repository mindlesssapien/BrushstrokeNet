import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from ml.config import NSTConfig
from ml.train import run_nst
from ml.utils.image_utils import open_image

log = logging.getLogger("brushstrokenet.jobs")


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"


@dataclass
class Job:
    id: str
    cfg: NSTConfig
    status: JobStatus = JobStatus.queued
    progress: float = 0.0
    error: Optional[str] = None
    stats: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    result_path: Optional[Path] = None


class JobManager:
    """In-process job queue with one GPU worker thread.

    The HTTP handler only validates input and enqueues; the event loop never runs model code.
    One worker serializes GPU work, so concurrent requests can't OOM the card.
    For multi-process / multi-host scale, swap this class for Celery/RQ/arq on Redis:
    the route code stays the same.
    """

    def __init__(self, model, output_dir: Path, max_queue: int = 32, ttl_seconds: int = 3600):
        self.model = model
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = ttl_seconds
        self.jobs: dict[str, Job] = {}
        self.q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="nst-worker", daemon=True)

    def start(self):
        self._worker.start()

    def stop(self):
        self._stop.set()
        self.q.put(None)
        self._worker.join(timeout=5)

    def submit(self, content: bytes, style: bytes, cfg: NSTConfig) -> Job:
        # decode now so bad uploads fail the request with 4xx, not later in the worker
        content_img, style_img = open_image(content), open_image(style)
        job = Job(id=uuid.uuid4().hex, cfg=cfg)  # server-generated name: no path traversal, no collisions
        with self._lock:
            self.jobs[job.id] = job
        try:
            self.q.put_nowait((job, content_img, style_img))
        except queue.Full:
            with self._lock:
                del self.jobs[job.id]
            raise
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self.jobs.get(job_id)

    def _run(self):
        while not self._stop.is_set():
            item = self.q.get()
            if item is None:
                break
            job, content_img, style_img = item
            job.status = JobStatus.running

            def progress(step, total, _losses):
                job.progress = round(step / total, 3)

            try:
                img, _history, stats = run_nst(self.model, content_img, style_img, job.cfg, progress)
                path = self.output_dir / f"{job.id}.png"
                img.save(path)
                job.result_path, job.stats = path, stats
                job.progress, job.status = 1.0, JobStatus.succeeded
                log.info("job %s done in %ss", job.id, stats["seconds"])
            except Exception as e:  # keep the worker alive
                log.exception("job %s failed", job.id)
                job.status, job.error = JobStatus.failed, str(e)
            finally:
                self._evict_old()

    def _evict_old(self):
        cutoff = time.time() - self.ttl
        with self._lock:
            old = [j for j in self.jobs.values() if j.created_at < cutoff and j.status in (JobStatus.succeeded, JobStatus.failed)]
            for j in old:
                if j.result_path:
                    j.result_path.unlink(missing_ok=True)
                del self.jobs[j.id]
