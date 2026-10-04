import os

# random VGG weights + tiny images keep tests offline and fast
os.environ.setdefault("NST_PRETRAINED", "0")
os.environ.setdefault("NST_IMAGE_SIZE", "64")
os.environ.setdefault("NST_STEPS", "5")
