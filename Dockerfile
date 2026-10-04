# One Dockerfile, two targets:
#   CPU: docker build -t brushstrokenet:cpu .
#   GPU: docker build -t brushstrokenet:gpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu124 .
# PyTorch's CUDA wheels bundle the CUDA runtime + cuDNN, so the base image stays python:slim;
# the host only needs the NVIDIA driver and nvidia-container-toolkit (`docker run --gpus all`).
FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/models \
    NST_OUTPUT_DIR=/data/outputs

WORKDIR /app

# 1) heavy, rarely-changing layer: torch from the chosen index (CPU or CUDA)
RUN pip install torch torchvision --index-url ${TORCH_INDEX}

# 2) remaining deps: only invalidated when pyproject.toml changes, not on code edits
COPY pyproject.toml ./
RUN python -c "import tomllib; print('\\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/requirements.txt \
 && pip install -r /tmp/requirements.txt

# 3) bake VGG-19 weights into the image: no 550 MB download on cold start, works offline
RUN python -c "from torchvision.models import vgg19, VGG19_Weights; vgg19(weights=VGG19_Weights.IMAGENET1K_V1)"

# 4) source last, so code changes rebuild in seconds
COPY app ./app
COPY ml ./ml

RUN useradd --create-home --uid 1000 nst && mkdir -p /data/outputs && chown -R nst /data /models
USER nst

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# one worker per GPU: each process holds its own copy of the model in VRAM
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
