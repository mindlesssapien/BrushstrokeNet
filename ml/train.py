import time
from typing import Callable, Optional

import torch

from ml import config
from ml.config import NSTConfig
from ml.models.vgg import VGG
from ml.utils.image_utils import to_image, to_tensor
from ml.utils.loss_utils import content_loss, gram_matrix, style_loss, total_variation_loss

torch.backends.cudnn.benchmark = True  # input shape is fixed during one optimization


class EarlyStop(Exception):
    pass


def run_nst(
    model: VGG,
    content_img,
    style_img,
    cfg: Optional[NSTConfig] = None,
    progress: Optional[Callable[[int, int, dict], None]] = None,
):
    """Gatys et al. style transfer. Returns (PIL image, loss history, stats).

    `model` is created once by the caller and reused across requests.
    `progress(step, total_steps, losses)` is called after every loss evaluation.
    """
    cfg = cfg or NSTConfig()
    device = next(model.parameters()).device
    torch.manual_seed(cfg.seed)

    content = to_tensor(content_img, cfg.image_size, device)
    style = to_tensor(style_img, cfg.image_size, device)

    # targets are constants: compute once, without building an autograd graph
    with torch.no_grad():
        _, content_feats = model(content)
        style_feats, _ = model(style)
        style_grams = {k: gram_matrix(v) for k, v in style_feats.items()}

    if cfg.init == "noise":
        target = torch.randn_like(content)
    else:
        target = content.clone()
    target.requires_grad_(True)

    if cfg.optimizer == "lbfgs":
        opt = torch.optim.LBFGS([target], max_iter=cfg.steps, line_search_fn="strong_wolfe")
    elif cfg.optimizer == "adam":
        opt = torch.optim.Adam([target], lr=cfg.adam_lr)
    else:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")

    history = []
    start = time.perf_counter()

    def closure():
        if len(history) >= cfg.steps:
            raise EarlyStop
        opt.zero_grad()
        t_style, t_content = model(target)
        c = content_loss(t_content, content_feats, config.CONTENT_LAYERS)
        s = style_loss(t_style, style_grams, config.STYLE_LAYERS)
        tv = total_variation_loss(target)
        loss = cfg.content_weight * c + cfg.style_weight * s + cfg.tv_weight * tv
        loss.backward()

        entry = {"step": len(history) + 1, "content": c.item(), "style": s.item(), "total": loss.item()}
        history.append(entry)
        if progress:
            progress(entry["step"], cfg.steps, entry)
        # if _plateaued(history, cfg):
        #     raise EarlyStop
        return loss

    try:
        if cfg.optimizer == "lbfgs":
            # one call runs up to cfg.steps evaluations internally
            while len(history) < cfg.steps:
                opt.step(closure)
        else:
            while len(history) < cfg.steps:
                opt.step(closure)
    except EarlyStop:
        pass

    stats = {
        "evaluations": len(history),
        "seconds": round(time.perf_counter() - start, 2),
        "device": str(device),
        "optimizer": cfg.optimizer,
        "final": history[-1] if history else None,
    }
    return to_image(target), history, stats


# def _plateaued(history, cfg):
#     w = cfg.early_stop_window
#     if len(history) <= w:
#         return False
#     old, new = history[-w - 1]["total"], history[-1]["total"]
#     return (old - new) / max(abs(old), 1e-12) < cfg.early_stop_rel_tol
