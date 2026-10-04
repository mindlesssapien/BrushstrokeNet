import os
from dataclasses import dataclass, field

import torch


def _env(name, default, cast=str):
    return cast(os.getenv(name, default))


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# VGG-19 `features` indices: conv1_1=0, conv2_1=5, conv3_1=10, conv4_1=19, conv4_2=21, conv5_1=28
STYLE_LAYERS = {"0": 1.0, "5": 1.0, "10": 1.0, "19": 1.0, "28": 1.0}
CONTENT_LAYERS = {"21": 1.0}

# set NST_PRETRAINED=0 to use random weights (tests / offline CI)
PRETRAINED = _env("NST_PRETRAINED", "1") == "1"


@dataclass
class NSTConfig:
    image_size: int = _env("NST_IMAGE_SIZE", 512 if torch.cuda.is_available() else 256, int)
    steps: int = _env("NST_STEPS", 300, int)
    content_weight: float = _env("NST_CONTENT_WEIGHT", 1.0, float)
    style_weight: float = _env("NST_STYLE_WEIGHT", 1e6, float)
    tv_weight: float = _env("NST_TV_WEIGHT", 0.0, float)  # try ~1e-3 to smooth noise
    optimizer: str = _env("NST_OPTIMIZER", "lbfgs")  # "lbfgs" | "adam"
    adam_lr: float = _env("NST_ADAM_LR", 0.02, float)
    init: str = "content"  # "content" | "noise"
    early_stop_rel_tol: float = 1e-4  # stop when loss improves by < 0.01% over a window
    early_stop_window: int = 20
    seed: int = 0
    extra: dict = field(default_factory=dict)
