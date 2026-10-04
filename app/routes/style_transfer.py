import queue
from typing import Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse

from app.services import JobStatus
from ml.config import NSTConfig

router = APIRouter()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}


async def _read_upload(f: UploadFile) -> bytes:
    if f.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"{f.filename}: unsupported type {f.content_type}")
    data = await f.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"{f.filename}: larger than {MAX_UPLOAD_BYTES // 2**20} MB")
    return data


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(
    request: Request,
    content: UploadFile = File(...),
    style: UploadFile = File(...),
    style_weight: float = Form(1e6, gt=0),
    content_weight: float = Form(1.0, gt=0),
    steps: int = Form(300, ge=1, le=1000),
    optimizer: Literal["lbfgs", "adam"] = Form("lbfgs"),
    image_size: Optional[int] = Form(None, ge=64, le=1024),
):
    cfg = NSTConfig(style_weight=style_weight, content_weight=content_weight, steps=steps, optimizer=optimizer)
    if image_size:
        cfg.image_size = image_size
    jobs = request.app.state.jobs
    try:
        job = jobs.submit(await _read_upload(content), await _read_upload(style), cfg)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except queue.Full:
        raise HTTPException(503, "queue full, retry later", headers={"Retry-After": "30"})
    return {"job_id": job.id, "status": job.status, "status_url": f"/jobs/{job.id}"}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request):
    job = request.app.state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    body = {"job_id": job.id, "status": job.status, "progress": job.progress}
    if job.status == JobStatus.succeeded:
        body["result_url"] = f"/jobs/{job.id}/result"
        body["stats"] = job.stats
    if job.status == JobStatus.failed:
        body["error"] = job.error
    return body


@router.get("/jobs/{job_id}/result", response_class=FileResponse)
def get_result(job_id: str, request: Request):
    job = request.app.state.jobs.get(job_id)
    if not job:
        raise HTTPException(404, "job not found")
    if job.status != JobStatus.succeeded:
        raise HTTPException(409, f"job is {job.status.value}")
    return FileResponse(job.result_path, media_type="image/png", filename=f"brushstrokenet_{job.id}.png")
