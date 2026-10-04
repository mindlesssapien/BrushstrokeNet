"""Adam vs L-BFGS and style-weight sweep on one content/style pair.

    uv run python -m experiments.compare_optimizers \
        --content data/content/content_image_1.png --style data/style/style_image_1.png

Writes outputs/experiments/{results.csv, loss_curves.png, grid.png, <run>.png}.
These are the numbers to quote for the "evaluated optimization trajectories" bullet.
"""
import argparse
import csv
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from ml.config import DEVICE, NSTConfig
from ml.models.vgg import VGG
from ml.utils.image_utils import open_image


def main():
    from ml.train import run_nst

    p = argparse.ArgumentParser()
    p.add_argument("--content", required=True)
    p.add_argument("--style", required=True)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--out", default="outputs/experiments")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    content = open_image(Path(args.content).read_bytes())
    style = open_image(Path(args.style).read_bytes())
    model = VGG().to(DEVICE)

    base = NSTConfig(steps=args.steps) #early_stop_rel_tol=0)  # fixed budget: fair comparison
    runs = {
        "lbfgs_sw1e6": replace(base, optimizer="lbfgs", style_weight=1e6),
        "adam_lr0.02_sw1e6": replace(base, optimizer="adam", adam_lr=0.02, style_weight=1e6),
        "adam_lr0.1_sw1e6": replace(base, optimizer="adam", adam_lr=0.1, style_weight=1e6),
        "lbfgs_sw1e4": replace(base, optimizer="lbfgs", style_weight=1e4),
        "lbfgs_sw1e5": replace(base, optimizer="lbfgs", style_weight=1e5),
        "lbfgs_sw1e7": replace(base, optimizer="lbfgs", style_weight=1e7),
    }

    rows, curves, images = [], {}, {}
    for name, cfg in runs.items():
        img, hist, stats = run_nst(model, content, style, cfg)
        img.save(out / f"{name}.png")
        images[name], curves[name] = img, hist
        f = stats["final"]
        rows.append({"run": name, "evals": stats["evaluations"], "seconds": stats["seconds"],
                     "content": f["content"], "style": f["style"], "total": f["total"]})
        print(rows[-1])

    with open(out / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for name in ("lbfgs_sw1e6", "adam_lr0.02_sw1e6", "adam_lr0.1_sw1e6"):
        ax[0].plot([h["total"] for h in curves[name]], label=name)
    ax[0].set(yscale="log", xlabel="loss evaluations", title="Total loss: L-BFGS vs Adam")
    for name in ("lbfgs_sw1e4", "lbfgs_sw1e5", "lbfgs_sw1e6", "lbfgs_sw1e7"):
        h = curves[name][-1]
        ax[1].scatter(h["content"], h["style"], label=name)
    ax[1].set(xscale="log", yscale="log", xlabel="final content loss", ylabel="final style loss",
              title="Content/style trade-off vs style weight")
    for a in ax:
        a.legend(); a.grid(True)
    fig.savefig(out / "loss_curves.png", dpi=120)

    w, h = images["lbfgs_sw1e6"].size
    grid = Image.new("RGB", (w * len(images), h), "white")
    for i, img in enumerate(images.values()):
        grid.paste(img.resize((w, h)), (i * w, 0))
    grid.save(out / "grid.png")


if __name__ == "__main__":
    main()
