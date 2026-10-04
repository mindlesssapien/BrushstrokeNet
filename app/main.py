import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from app import routes
from app.services import JobManager
from ml.config import DEVICE
from ml.models.vgg import VGG

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load VGG-19 once per process, not once per request
    model = VGG().to(DEVICE)
    app.state.jobs = JobManager(model, Path(os.getenv("NST_OUTPUT_DIR", "outputs/jobs")))
    app.state.jobs.start()
    yield
    app.state.jobs.stop()


app = FastAPI(title="BrushstrokeNet API", lifespan=lifespan)
app.include_router(routes.nst_router)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "queued": app.state.jobs.q.qsize()}
