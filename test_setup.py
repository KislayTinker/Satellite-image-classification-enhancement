import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

print("=" * 50)
print("ENVIRONMENT CHECK")
print("=" * 50)

print(f"NumPy version: {np.__version__}")
print(f"OpenCV version: {cv2.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA (GPU) available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print("Running on CPU only (training will be slower, but Part 1-4 work fine without GPU)")

print("=" * 50)
print("If you see this message with no errors above, setup is successful!")
print("=" * 50)