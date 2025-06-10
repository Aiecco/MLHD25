import numpy as np
import random
import time
import sys
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm  # For progress bar
from typing import Optional, Tuple

# Constants
OUTPUT_SIZE: int = 500
OFFSET_PERCENT: int = 5  # cropping offset around the hand

BLUR_KERNEL_SIZE: int = 7 # kernel size for Gaussian Blur
MIN_CONTOUR_AREA_FACTOR: float = 0.05 # minimum contour area relative to image area

# ---Functions
def apply_clahe(image_gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image_gray)

def detect_and_crop_hand_cv(
    image_gray: np.ndarray,
    offset_percent: int = OFFSET_PERCENT,
    blur_ksize: int = BLUR_KERNEL_SIZE,
    min_area_factor: float = MIN_CONTOUR_AREA_FACTOR) -> Optional[np.ndarray]:

    h, w = image_gray.shape

    # blur to reduce noise
    blurred = cv2.GaussianBlur(image_gray, (blur_ksize, blur_ksize), 0)

    # thresholding (cv2 automatically finds a threshold)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # find confines
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    # find the largest contour (assuming it's the hand)
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    min_area = h * w * min_area_factor

    if area < min_area:
        return None

    x, y, wb, hb = cv2.boundingRect(largest_contour) # get Bounding Box

    # calculate offset and new bounding box coordinates
    offset_h = int((h * offset_percent) / 100)
    offset_w = int((w * offset_percent) / 100)

    y_min = max(0, y - offset_h)
    y_max = min(h, y + hb + offset_h)
    x_min = max(0, x - offset_w)
    x_max = min(w, x + wb + offset_w)

    # crop the original img
    if y_max > y_min and x_max > x_min:
        return image_gray[y_min:y_max, x_min:x_max]

def resize(image_gray: np.ndarray, output_size: int = OUTPUT_SIZE) -> np.ndarray:
    """resize img to 500x500 but preserving aspect ratio by padding."""
    h, w = image_gray.shape
    max_dim = max(h, w)

    # Calculate padding
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad

    # pad with mean pixel value
    padding = int(np.mean(image_gray))
    padded_img = cv2.copyMakeBorder(image_gray, top_pad, bottom_pad, left_pad, right_pad,
                                    cv2.BORDER_CONSTANT, value=padding)

    # resize to final output size
    resized_img = cv2.resize(padded_img, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return resized_img


def aug(image: np.ndarray) -> np.ndarray:
    """
    Applies a set of random augmentations to the image.
    Includes: horizontal flip, rotations, inversion, brightness, contrast.
    """
    # Random horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # Random 90/180/270 deg rotation
    rot_code = random.choice([None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
    if rot_code is not None:
        image = cv2.rotate(image, rot_code)

    # Random inversion (negative image)
    if random.random() > 0.5:
        image = 255 - image  # Invert pixel values for 8-bit grayscale

    # Random brightness adjustment
    if random.random() > 0.5:
        alpha = random.uniform(0.7, 1.3)  # Factor for brightness (1.0 is no change)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

    # Random contrast adjustment
    if random.random() > 0.5:
        alpha = random.uniform(0.7, 1.3)  # Factor for contrast
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

    return image
