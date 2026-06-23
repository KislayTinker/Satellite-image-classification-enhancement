# src/test_model.py
# Purpose: Sanity-check that our CNN model accepts satellite images
# and produces correctly-shaped output, BEFORE we attempt real training.

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from model import SatelliteCNN
from dataset_split import create_data_splits

# Step 1: Create the model
model = SatelliteCNN(num_classes=10)
print("Model created successfully!")

# Step 2: Count total parameters (the "knowledge capacity" of the model)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {total_params:,}")

# Step 3: Load a real batch of data and pass it through the model
train_ds, val_ds, test_ds = create_data_splits()

# Grab 4 sample images and stack them into a batch
images = torch.stack([train_ds[i][0] for i in range(4)])
labels = torch.tensor([train_ds[i][1] for i in range(4)])

print(f"\nInput batch shape: {images.shape}")  # should be [4, 3, 64, 64]

# Step 4: Pass images through the model (forward pass)
with torch.no_grad():  # we're not training yet, so no need to track gradients
    output = model(images)

print(f"Output shape: {output.shape}")  # should be [4, 10] - 4 images, 10 class scores each
print(f"\nSample raw output for first image:\n{output[0]}")
print(f"\nTrue labels for this batch: {labels.tolist()}")
print(f"Class names: {[train_ds.dataset.classes[l] for l in labels.tolist()]}")

# Step 5: Check if GPU is being used (relevant for Part 6 training speed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice available for training: {device}")