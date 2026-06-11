# BrushstrokeNet

BrushstrokeNet is a PyTorch-based implementation of the seminal paper "A Neural Algorithm of Artistic Style" by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

This project uses a Deep Convolutional Neural Network (VGG-19) to blend the semantic content of a Content Image with the stylistic features of a Style Image, allowing you to generate high-quality stylized artwork.

---

## Features

* **VGG-19 Architecture:** Leverages pre-trained ImageNet weights for high-fidelity feature extraction.
* **Customizable Weighting:** Fine-tune the balance between content preservation and style intensity.
* **Gram Matrix Optimization:** Accurately captures texture, color, and brushstroke patterns.
* **L-BFGS & Adam Support:** Toggle between optimizer algorithms for speed vs. precise detail.
* **Intermediate Saving:** Automatically saves progress images so you can watch the artwork evolve.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/BrushstrokeNet.git
cd BrushstrokeNet

# Create the environment and install all dependencies from pyproject.toml
uv sync
```

## How It Works

The network optimizes a blank or noise image (or a copy of the content image) by minimizing a joint loss function:

$$L_{total} = \alpha L_{content} + \beta L_{style}$$

* **Content Loss ($L_{content}$):** Measures the Mean Squared Error (MSE) between the feature representations of the content image and the generated image at a deep layer (e.g., `conv4_2`).
* **Style Loss ($L_{style}$):** Computes the MSE between the Gram Matrices (feature correlations) of the style image and the generated image across multiple layers (e.g., `conv1_1` through `conv5_1`).

---

## Acknowledgments & References

* Gatys, L. A., Ecker, A. S., & Bethge, M. (2015). *A Neural Algorithm of Artistic Style*. [arXiv:1508.06576](https://www.google.com/search?q=https%3A%2F%2Farxiv.org%2Fabs%2F1508.06576).
---
