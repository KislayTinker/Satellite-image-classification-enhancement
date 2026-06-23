# src/visualize_predictions.py
# Purpose: Show real test images alongside the model's predictions vs true labels.
# Green title = correct prediction, Red title = incorrect prediction.

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import matplotlib.pyplot as plt
import numpy as np

from model import SatelliteCNN
from dataset_split import create_data_splits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds, val_ds, test_ds = create_data_splits()
class_names = train_ds.dataset.classes

model = SatelliteCNN(num_classes=10).to(device)
model.load_state_dict(torch.load("models/best_satellite_cnn.pth"))
model.eval()

# Pick 8 random test images
indices = np.random.choice(len(test_ds), 8, replace=False)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

# Mean/std used earlier for normalization - we need to REVERSE it to display images properly
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

for idx, ax in zip(indices, axes):
    image, true_label = test_ds[idx]
    
    with torch.no_grad():
        output = model(image.unsqueeze(0).to(device))  # add batch dimension
        predicted_label = torch.argmax(output, dim=1).item()
    
    # Reverse normalization to display the image correctly
    image_display = image.permute(1, 2, 0).numpy()  # change from (C,H,W) to (H,W,C)
    image_display = image_display * IMAGENET_STD + IMAGENET_MEAN
    image_display = np.clip(image_display, 0, 1)
    
    ax.imshow(image_display)
    
    is_correct = predicted_label == true_label
    color = "green" if is_correct else "red"
    
    ax.set_title(
        f"True: {class_names[true_label]}\nPred: {class_names[predicted_label]}",
        color=color, fontsize=10
    )
    ax.axis("off")

plt.suptitle("Model Predictions on Test Images (Green=Correct, Red=Wrong)", fontsize=14)
plt.tight_layout()
plt.savefig("outputs/predictions/sample_predictions.png", dpi=150)
print("Saved to outputs/predictions/sample_predictions.png")
plt.show()