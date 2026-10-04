import torch

from ml.models.vgg import VGG
from ml.train import run_nst
from ml.utils.loss_utils import gram_matrix
from PIL import Image


def test_gram_is_batched_symmetric_and_size_normalized():
    x = torch.randn(2, 8, 5, 7)
    g = gram_matrix(x)
    assert g.shape == (2, 8, 8)
    assert torch.allclose(g, g.transpose(1, 2))
    # doubling spatial size with the same statistics should not change the Gram scale much
    assert torch.allclose(gram_matrix(x.repeat(1, 1, 2, 1)), g, atol=1e-5)


def test_vgg_is_frozen_and_truncated():
    m = VGG(pretrained=False)
    assert not any(p.requires_grad for p in m.parameters())
    assert len(m.vgg) == 29  # up to and including conv5_1


def test_run_nst_survives_many_lbfgs_evaluations():
    # regression: the original loop crashed on the 2nd closure call
    m = VGG(pretrained=False)
    c = Image.new("RGB", (80, 64), "red")
    s = Image.new("RGB", (64, 64), "blue")
    from ml.config import NSTConfig
    img, hist, stats = run_nst(m, c, s, NSTConfig(image_size=64, steps=10)) #early_stop_rel_tol=0
    assert stats["evaluations"] == 10
    assert img.size == (80, 64)  # aspect ratio and content size preserved
