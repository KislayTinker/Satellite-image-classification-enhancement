# src/dataset_split.py
# Purpose: Split the full EuroSAT dataset into Train/Validation/Test sets
# and apply the correct preprocessing transform to each.

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torchvision.datasets import EuroSAT
from torch.utils.data import random_split
import torch

from preprocessing import get_train_transform, get_eval_transform

def create_data_splits(root="data/raw", seed=42):
    """
    Loads EuroSAT and splits it into train/val/test.
    
    Returns:
        train_dataset, val_dataset, test_dataset
    """
    # Step 1: Load the dataset WITHOUT any transform first
    # (we'll apply transforms separately after splitting)
    full_dataset = EuroSAT(root=root, download=False)

    # Step 2: Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size  # remainder, avoids rounding errors

    print(f"Total images: {total_size}")
    print(f"Train: {train_size} | Validation: {val_size} | Test: {test_size}")

    # Step 3: Randomly split, using a fixed seed for reproducibility
    # (seed ensures you get the SAME split every time you run this -
    #  important so your results are consistent and comparable across runs)
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # Step 4: Apply the correct transform to each split
    train_dataset.dataset.transform = get_train_transform()
    val_dataset.dataset.transform = get_eval_transform()
    test_dataset.dataset.transform = get_eval_transform()

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    train_ds, val_ds, test_ds = create_data_splits()
    
    print("\n--- Verification ---")
    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Test dataset size: {len(test_ds)}")
    
    # Grab one sample and check its shape after preprocessing
    sample_image, sample_label = train_ds[0]
    print(f"\nSample image tensor shape: {sample_image.shape}")  # should be [3, 64, 64]
    print(f"Sample image min value: {sample_image.min():.3f}")   # should be negative (due to normalization)
    print(f"Sample image max value: {sample_image.max():.3f}")   # should be positive