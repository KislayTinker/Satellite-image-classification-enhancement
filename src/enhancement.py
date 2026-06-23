# src/enhancement.py
# Purpose: Apply classical image enhancement techniques to satellite images:
# histogram equalization (contrast), sharpening (clarity), and super-resolution (upscaling).
# These functions take a raw image and return an improved version.

import cv2
import numpy as np


def apply_histogram_equalization(image):
    """
    Improves contrast by redistributing pixel brightness values.
    
    Args:
        image: numpy array, RGB image with values 0-255, shape (H, W, 3)
    
    Returns:
        Enhanced image with improved contrast, same shape as input
    """
    # Convert RGB to YCrCb color space.
    # WHY: We only want to equalize BRIGHTNESS, not color.
    # YCrCb separates brightness (Y channel) from color information (Cr, Cb channels).
    # If we equalized R, G, B channels separately, colors would shift unnaturally.
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    
    # Apply histogram equalization ONLY to the Y (brightness) channel
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    
    # Convert back to RGB for normal viewing/saving
    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    
    return enhanced


def apply_sharpening(image):
    """
    Sharpens image edges using a convolution kernel.
    
    Args:
        image: numpy array, RGB image with values 0-255, shape (H, W, 3)
    
    Returns:
        Sharpened image, same shape as input
    """
    # Define the sharpening kernel (explained in theory section above)
    kernel = np.array([
        [0, -1,  0],
        [-1, 5, -1],
        [0, -1,  0]
    ])
    
    # cv2.filter2D "slides" this kernel across every pixel in the image
    sharpened = cv2.filter2D(image, -1, kernel)
    
    return sharpened


def apply_super_resolution(image, scale_factor=2):
    """
    Upscales image resolution using bicubic interpolation.
    
    Args:
        image: numpy array, RGB image, shape (H, W, 3)
        scale_factor: how much bigger to make the image (2 = double size)
    
    Returns:
        Upscaled image, shape (H*scale_factor, W*scale_factor, 3)
    """
    height, width = image.shape[:2]
    new_height, new_width = height * scale_factor, width * scale_factor
    
    # INTER_CUBIC = bicubic interpolation, considered the best classical
    # method for smooth, natural-looking upscaling (vs INTER_LINEAR which is faster but blurrier)
    upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    return upscaled


def enhance_pipeline(image):
    """
    Runs the full enhancement pipeline: equalize -> sharpen -> upscale.
    This is the main function we'll call from other scripts.
    
    Args:
        image: numpy array, RGB image, shape (H, W, 3)
    
    Returns:
        Fully enhanced and upscaled image
    """
    step1 = apply_histogram_equalization(image)
    step2 = apply_sharpening(step1)
    step3 = apply_super_resolution(step2, scale_factor=2)
    
    return step3