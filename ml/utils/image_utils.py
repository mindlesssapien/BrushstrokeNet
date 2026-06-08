from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 512 if torch.cuda.is_available() else 128

loader = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize(imsize),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

unloader = transforms.ToPILImage()

def image_loader(image_path):
    image = Image.open(image_path)
    image = loader(image).unsqueeze(0)
    return image.to(device)

def img_show(tensor, title=None):
    image = tensor.cpu().clone()
    denormalization = transforms.Normalize((-2.12, -2.04, -1.80),
                                           (4.37, 4.46, 4.44))
    image = image.squeeze(0)
    image = denormalization(image).clamp(0, 1)
    image = unloader(image)
    plt.imshow(image)
    if title:
        plt.title(title)