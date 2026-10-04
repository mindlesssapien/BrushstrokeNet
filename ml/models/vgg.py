import torch.nn as nn
from torchvision.models import VGG19_Weights, vgg19

from ml import config


class VGG(nn.Module):
    """Frozen VGG-19 trunk that returns the activations needed for NST.

    - truncated after the deepest layer we use (no wasted conv5_2..pool5 compute)
    - parameters frozen, so backward only computes gradients w.r.t. the image
    - optional avg-pooling (Gatys et al. report smoother results than max-pooling)
    """

    def __init__(self, style_layers=None, content_layers=None, avg_pool=False, pretrained=None):
        super().__init__()
        self.style_layers = list(style_layers or config.STYLE_LAYERS)
        self.content_layers = list(content_layers or config.CONTENT_LAYERS)
        taps = self.style_layers + self.content_layers
        last = max(int(i) for i in taps)

        pretrained = config.PRETRAINED if pretrained is None else pretrained
        weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
        features = vgg19(weights=weights).features[: last + 1]

        layers = []
        for layer in features:
            if isinstance(layer, nn.ReLU):
                # in-place ReLU would overwrite the conv outputs we collect
                layer = nn.ReLU(inplace=False)
            elif avg_pool and isinstance(layer, nn.MaxPool2d):
                layer = nn.AvgPool2d(kernel_size=2, stride=2)
            layers.append(layer)
        self.vgg = nn.Sequential(*layers)

        self.vgg.requires_grad_(False)
        self.eval()

    def forward(self, x):
        style, content = {}, {}
        for name, layer in self.vgg.named_children():
            x = layer(x)
            if name in self.style_layers:
                style[name] = x
            if name in self.content_layers:
                content[name] = x
        return style, content
