# src/train.py
# Purpose: Train our CNN model on the EuroSAT dataset.
# This is the core "learning" step of the project.

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time

from model import SatelliteCNN
from dataset_split import create_data_splits


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Runs ONE full pass through the training data.
    Returns: average loss and accuracy for this epoch.
    """
    model.train()  # tells PyTorch "we are training" (activates Dropout, etc.)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Step 1: Clear old gradients (PyTorch accumulates them by default)
        optimizer.zero_grad()
        
        # Step 2: Forward pass - get model's predictions
        outputs = model(images)
        
        # Step 3: Calculate how wrong the predictions were
        loss = criterion(outputs, labels)
        
        # Step 4: Backward pass - calculate which direction to adjust each weight
        loss.backward()
        
        # Step 5: Optimizer applies the adjustment
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)  # pick the class with highest score
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    
    return epoch_loss, epoch_accuracy


def evaluate(model, val_loader, criterion, device):
    """
    Checks model performance on validation data.
    NO weight updates happen here - we're just measuring, not learning.
    """
    model.eval()  # tells PyTorch "we are evaluating" (deactivates Dropout, etc.)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # don't waste memory/time tracking gradients - we're not training
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_accuracy = correct / total
    
    return epoch_loss, epoch_accuracy


def main():
    # ----- Setup -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # Load data
    train_ds, val_ds, test_ds = create_data_splits()
    
    # DataLoader handles batching and shuffling automatically
    # batch_size=32 means we process 32 images at once before updating weights
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    
    # Model, loss function, optimizer
    model = SatelliteCNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 15
    best_val_accuracy = 0.0
    
    print(f"\nStarting training for {num_epochs} epochs...\n")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2%} | "
              f"Time: {elapsed:.1f}s")
        
        # Save the model only when it improves - this is called "checkpointing"
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), "models/best_satellite_cnn.pth")
            print(f"  --> New best model saved! (Val Acc: {val_acc:.2%})")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_accuracy:.2%}")


if __name__ == "__main__":
    main()