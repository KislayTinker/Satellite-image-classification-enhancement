# src/explore_data.py
# Purpose: Load a few sample images from the dataset and visualize them.
# This helps us "see" what kind of data we're working with before building any models.

from torchvision.datasets import EuroSAT
import matplotlib.pyplot as plt

# Load the dataset (already downloaded, so download=False this time)
dataset = EuroSAT(root="data/raw", download=False)

# Print basic info
print(f"Total number of images: {len(dataset)}")
print(f"Classes: {dataset.classes}")

# Let's look at the very first image in the dataset
image, label = dataset[0]
print(f"\nFirst image's class: {dataset.classes[label]}")
print(f"Image size: {image.size}")  # (width, height) in pixels
print(f"Image type: {type(image)}")

# Visualize 5 random images, one from different classes
fig, axes = plt.subplots(1, 5, figsize=(15, 3))

for i in range(5):
    image, label = dataset[i * 1000]  # pick spaced-out samples to get variety
    axes[i].imshow(image)
    axes[i].set_title(dataset.classes[label])
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("outputs/sample_images.png")  # save instead of just showing, so we have proof it worked
print("\nSaved sample images to outputs/sample_images.png")
plt.show()