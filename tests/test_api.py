import io
import time

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


def _png(color):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format="PNG")
    return buf.getvalue()


def test_job_lifecycle():
    with TestClient(app) as client:
        r = client.post("/jobs", files={"content": ("c.png", _png("red"), "image/png"),
                                        "style": ("s.png", _png("blue"), "image/png")},
                        data={"steps": "5"})
        assert r.status_code == 202, r.text
        job_id = r.json()["job_id"]

        for _ in range(100):
            body = client.get(f"/jobs/{job_id}").json()
            if body["status"] in ("succeeded", "failed"):
                break
            time.sleep(0.1)
        assert body["status"] == "succeeded", body

        img = client.get(body["result_url"])
        assert img.status_code == 200
        assert img.headers["content-type"] == "image/png"
        Image.open(io.BytesIO(img.content)).verify()


def test_rejects_non_images_and_path_tricks():
    with TestClient(app) as client:
        r = client.post("/jobs", files={"content": ("../../app/main.py", b"not an image", "image/png"),
                                        "style": ("s.png", _png("blue"), "image/png")})
        assert r.status_code == 422
        r = client.post("/jobs", files={"content": ("c.txt", b"hi", "text/plain"),
                                        "style": ("s.png", _png("blue"), "image/png")})
        assert r.status_code == 415


def test_health_responds_while_job_runs():
    with TestClient(app) as client:
        client.post("/jobs", files={"content": ("c.png", _png("red"), "image/png"),
                                    "style": ("s.png", _png("blue"), "image/png")}, data={"steps": "50"})
        assert client.get("/health").status_code == 200
