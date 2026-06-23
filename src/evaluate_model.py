# src/evaluate_model.py
# Purpose: Load our best trained model and evaluate it fairly on the TEST set
# (images the model has never seen or been tuned against).
# Also generates a confusion matrix to see which classes get confused with each other.

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np

from model import SatelliteCNN
from dataset_split import create_data_splits


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test data
    train_ds, val_ds, test_ds = create_data_splits()
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    class_names = train_ds.dataset.classes
    
    # Load our best saved model
    model = SatelliteCNN(num_classes=10).to(device)
    model.load_state_dict(torch.load("models/best_satellite_cnn.pth"))
    model.eval()
    
    print("Model loaded. Evaluating on TEST set (never seen during training)...\n")
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Overall test accuracy
    test_accuracy = (all_predictions == all_labels).mean()
    print(f"FINAL TEST ACCURACY: {test_accuracy:.2%}\n")
    
    # Detailed per-class report (precision, recall, F1-score)
    print("Per-Class Performance Report:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # Build and visualize the confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title(f'Confusion Matrix - Test Accuracy: {test_accuracy:.2%}')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig("outputs/predictions/confusion_matrix.png", dpi=150)
    print("\nSaved confusion matrix to outputs/predictions/confusion_matrix.png")
    plt.show()


if __name__ == "__main__":
    main()