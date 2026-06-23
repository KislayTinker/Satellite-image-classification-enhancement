# src/model.py
# Purpose: Define the CNN architecture that will learn to classify
# satellite images into 10 land-use categories.

import torch
import torch.nn as nn


class SatelliteCNN(nn.Module):
    """
    A simple CNN for classifying 64x64 satellite images into 10 classes.
    
    Architecture: 3 blocks of (Conv -> ReLU -> MaxPool), then Flatten -> Linear -> Output
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # ----- Block 1 -----
        # Input: 3 channels (RGB), Output: 32 "feature maps" (learned pattern detectors)
        # kernel_size=3 means each filter looks at a 3x3 patch of pixels at a time
        # padding=1 keeps the image size the same after convolution (64x64 stays 64x64)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)  # 64x64 -> 32x32
        
        # ----- Block 2 -----
        # Takes the 32 feature maps from Block 1, produces 64 more refined feature maps
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)  # 32x32 -> 16x16
        
        # ----- Block 3 -----
        # Takes 64 feature maps, produces 128 even more refined feature maps
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2)  # 16x16 -> 8x8
        
        # ----- Flatten + Fully Connected -----
        # After 3 pooling layers: 64x64 -> 32x32 -> 16x16 -> 8x8
        # We have 128 feature maps, each 8x8 in size
        # Flattened size = 128 * 8 * 8 = 8192 numbers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=128 * 8 * 8, out_features=256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # randomly "turns off" 30% of neurons during training
                                          # this prevents overfitting (explained below)
        self.fc2 = nn.Linear(in_features=256, out_features=num_classes)
    
    def forward(self, x):
        """
        Defines how data flows through the network.
        x: input tensor of shape (batch_size, 3, 64, 64)
        Returns: output tensor of shape (batch_size, 10) - raw scores for each class
        """
        # Block 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # Flatten and classify
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x  # raw scores (called "logits") - not probabilities yet