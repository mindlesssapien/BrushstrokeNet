import io

import torch
import torchvision.transforms as T
from PIL import Image

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def open_image(data: bytes) -> Image.Image:
    """Decode and validate untrusted bytes. Raises ValueError on anything that isn't an image."""
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # cheap integrity check, invalidates the handle
        img = Image.open(io.BytesIO(data))
    except Exception as e:
        raise ValueError("not a valid image") from e
    return img.convert("RGB")  # drops alpha (RGBA PNGs) and palette modes


def to_tensor(img: Image.Image, size: int, device) -> torch.Tensor:
    tf = T.Compose([T.Resize(size), T.ToTensor()])  # resize shorter side, keep aspect ratio
    x = tf(img).unsqueeze(0)
    return ((x - MEAN) / STD).to(device)


def to_image(x: torch.Tensor) -> Image.Image:
    x = x.detach().cpu() * STD + MEAN  # exact inverse of the normalization
    return T.ToPILImage()(x.squeeze(0).clamp(0, 1))


def to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
