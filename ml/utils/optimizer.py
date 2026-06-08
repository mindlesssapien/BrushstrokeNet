import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image

def save(target, i):
    denormalization = transforms.Normalize((-2.12, -2.04, -1.80),
                                           (4.37, 4.46, 4.44))
    img = target.clone().squeeze()
    img = denormalization(img).clamp(0, 1)
    save_image(img, f'./outputs/result/result_{i}.png')

def adam_optimizer(x, lr):
    return optim.Adam([x], lr=lr)

def lbfgs_optimizer(x):
    return optim.LBFGS([x])