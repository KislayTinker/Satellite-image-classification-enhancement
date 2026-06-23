# src/test_enhancement.py
# Purpose: Load real satellite images and visually compare before vs after enhancement.
# This produces the kind of side-by-side image you'll want in your resume/GitHub README.

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from torchvision.datasets import EuroSAT

from enhancement import apply_histogram_equalization, apply_sharpening, apply_super_resolution, enhance_pipeline


def pil_to_numpy(pil_image):
    """Converts a PIL image (what EuroSAT gives us) into a numpy array OpenCV can use."""
    return np.array(pil_image)


# Load dataset (no transform needed - we want raw images for this test)
dataset = EuroSAT(root="data/raw", download=False)

# Pick one sample image to demonstrate on (index 5000 = some random Forest/River/etc image)
sample_image, label = dataset[5000]
print(f"Selected image class: {dataset.classes[label]}")

# Convert to numpy array for OpenCV processing
image_np = pil_to_numpy(sample_image)
print(f"Original image shape: {image_np.shape}")

# Apply each technique separately so we can compare all stages
equalized = apply_histogram_equalization(image_np)
sharpened = apply_sharpening(image_np)
upscaled = apply_super_resolution(image_np, scale_factor=2)
full_pipeline = enhance_pipeline(image_np)

print(f"Upscaled image shape: {upscaled.shape}")  # should be double the original size

# Create a before/after comparison figure
fig, axes = plt.subplots(1, 5, figsize=(20, 4))

axes[0].imshow(image_np)
axes[0].set_title("Original")
axes[0].axis("off")

axes[1].imshow(equalized)
axes[1].set_title("Histogram Equalized")
axes[1].axis("off")

axes[2].imshow(sharpened)
axes[2].set_title("Sharpened")
axes[2].axis("off")

axes[3].imshow(upscaled)
axes[3].set_title("Upscaled (2x)")
axes[3].axis("off")

axes[4].imshow(full_pipeline)
axes[4].set_title("Full Pipeline")
axes[4].axis("off")

plt.suptitle(f"Enhancement Comparison - {dataset.classes[label]} Image", fontsize=14)
plt.tight_layout()

# Save to outputs folder
plt.savefig("outputs/enhanced_images/comparison.png", dpi=150)
print("\nSaved comparison image to outputs/enhanced_images/comparison.png")

plt.show()