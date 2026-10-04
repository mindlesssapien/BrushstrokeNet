# BrushstrokeNet

BrushstrokeNet is a PyTorch-based implementation of the seminal paper "A Neural Algorithm of Artistic Style" by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

This project uses a Deep Convolutional Neural Network (VGG-19) to blend the semantic content of a Content Image with the stylistic features of a Style Image, allowing you to generate high-quality stylized artwork.

---

## Features

* **VGG-19 Architecture:** Leverages pre-trained ImageNet weights for high-fidelity feature extraction.
* **Customizable Weighting:** Fine-tune the balance between content preservation and style intensity.
* **Gram Matrix Optimization:** Accurately captures texture, color, and brushstroke patterns.
* **L-BFGS & Adam Support:** Toggle between optimizer algorithms for speed vs. precise detail.
* **Intermediate Saving:** Automatically saves progress images so you can watch the artwork evolve.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/BrushstrokeNet.git
cd BrushstrokeNet

# Create the environment and install all dependencies from pyproject.toml
uv sync
```

PyTorch implementation of Gatys et al., "A Neural Algorithm of Artistic Style", served as an asynchronous job API with FastAPI and packaged with Docker for CPU or GPU.

## How it works

The output image's pixels are the only trainable tensor. A frozen, truncated VGG-19 extracts features; the loss is

    L = α · MSE(F_conv4_2(x), F_conv4_2(content)) + β · Σ_l MSE(G_l(x), G_l(style)) + γ · TV(x)

where `G_l` is the size-normalized Gram matrix at conv1_1, conv2_1, conv3_1, conv4_1, conv5_1. L-BFGS (strong-Wolfe line search) or Adam minimizes it, with early stopping on plateau.

## Run

```bash
uv sync
uv run fastapi dev app/main.py           # http://localhost:8000/docs
# or
docker compose up --build api            # CPU
docker compose up --build api-gpu        # GPU (host needs nvidia-container-toolkit)
```

```bash
curl -F content=@data/content/content_image_1.png -F style=@data/style/style_image_1.png \
     -F style_weight=1e6 -F optimizer=lbfgs localhost:8000/jobs        # -> 202 {"job_id": ...}
curl localhost:8000/jobs/<job_id>                                       # status + progress
curl -o out.png localhost:8000/jobs/<job_id>/result                     # PNG
```

## Experiments

```bash
uv run python -m experiments.compare_optimizers \
  --content data/content/content_image_1.png --style data/style/style_image_1.png
```

Writes Adam vs L-BFGS loss curves, a style-weight sweep, and `results.csv` to `outputs/experiments/`.

## Tests

```bash
uv run pytest     # offline: uses random VGG weights and 64 px images
```

## Reference

Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). *A Neural Algorithm of Artistic Style*. arXiv:1508.06576.
