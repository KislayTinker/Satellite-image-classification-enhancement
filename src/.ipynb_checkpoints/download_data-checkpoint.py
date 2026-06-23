# src/download_data.py
# Purpose: Automatically download and organize the EuroSAT satellite dataset.
# This replaces manual zip downloading - torchvision handles everything for us.

from torchvision.datasets import EuroSAT

print("Downloading EuroSAT dataset... this may take a few minutes (dataset is ~89 MB for RGB version)")

# This single line does 3 things:
# 1. Downloads the dataset zip file
# 2. Extracts it automatically
# 3. Organizes it into class folders (Forest, River, Industrial, etc.)
dataset = EuroSAT(
    root="data/raw",      # where to save the data
    download=True          # tells torchvision to fetch it from the internet
)

print("Download complete!")
print(f"Total images in dataset: {len(dataset)}")
print(f"Class names: {dataset.classes}")