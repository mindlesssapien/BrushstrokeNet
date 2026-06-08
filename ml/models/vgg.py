import torch.nn as nn
import torchvision.models as models

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.select_features = ['0', '5', '10', '19', '28']
        self.vgg = models.vgg19(pretrained=True).features

    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select_features:
                features.append(x)
        return features