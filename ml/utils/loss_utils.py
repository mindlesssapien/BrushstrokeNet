import torch
import torch.nn.functional as F


def gram_matrix(x: torch.Tensor) -> torch.Tensor:
    """Batched, size-normalized Gram matrix: (B, C, H, W) -> (B, C, C)."""
    b, c, h, w = x.size()
    f = x.view(b, c, h * w)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


def content_loss(target_feats: dict, content_feats: dict, weights: dict) -> torch.Tensor:
    return sum(weights[k] * F.mse_loss(target_feats[k], content_feats[k]) for k in weights)


def style_loss(target_feats: dict, style_grams: dict, weights: dict) -> torch.Tensor:
    return sum(weights[k] * F.mse_loss(gram_matrix(target_feats[k]), style_grams[k]) for k in weights)


def total_variation_loss(x: torch.Tensor) -> torch.Tensor:
    """Penalizes neighbouring-pixel differences; suppresses high-frequency noise."""
    return (x[..., 1:, :] - x[..., :-1, :]).abs().mean() + (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
