# src/preprocessing.py
# Purpose: Define how raw satellite images get transformed into 
# a clean, consistent format that our model can learn from.

import torchvision.transforms as transforms

# These mean/std values come from ImageNet (a massive, well-studied photo dataset).
# Even though our images are satellite photos, RGB color statistics are similar enough
# that these values work well as a starting point. This is a very common practice.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transform():
    """
    Preprocessing pipeline used during TRAINING.
    Includes random flips/rotations to artificially create variety (data augmentation) -
    this helps the model generalize better and not just memorize exact images.
    """
    return transforms.Compose([
        transforms.Resize((64, 64)),              # ensure consistent size
        transforms.RandomHorizontalFlip(p=0.5),    # 50% chance to flip image left-right
        transforms.RandomVerticalFlip(p=0.5),      # 50% chance to flip image top-bottom
        transforms.RandomRotation(degrees=15),     # randomly rotate up to 15 degrees
        transforms.ToTensor(),                     # converts image (0-255) to a PyTorch tensor (0-1)
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),  # final normalization
    ])

def get_eval_transform():
    """
    Preprocessing pipeline used during VALIDATION/TESTING.
    No random flips/rotations here - we want consistent, repeatable results
    when checking how well our model actually performs.
    """
    return transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])