import os
import numpy as np
import random
import cv2
from tqdm import tqdm  # For progress bar
from typing import Optional, Tuple

# --- Constants ---
OUTPUT_SIZE: int = 500
"""
int: Default target dimension for resizing images (height and width).
     Images will be resized to (OUTPUT_SIZE, OUTPUT_SIZE) pixels.
"""
OFFSET_PERCENT: int = 5
"""
int: Percentage by which to expand the detected hand's bounding box
     during cropping, providing a margin around the hand.
"""
BLUR_KERNEL_SIZE: int = 7
"""
int: Kernel size for the Gaussian Blur applied to images during hand detection.
     Must be an odd integer.
"""
MIN_CONTOUR_AREA_FACTOR: float = 0.05
"""
float: Minimum contour area as a factor of the total image area.
       Contours smaller than this threshold are ignored during hand detection,
       to filter out small artifacts.
"""

# --- Functions ---
def apply_clahe(image_gray: np.ndarray) -> np.ndarray:
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to a grayscale image.

    CLAHE is used to enhance the contrast of the image, especially in regions
    where contrast might be low, while limiting noise amplification. This is
    particularly useful for medical images like X-rays.

    Args:
        image_gray (np.ndarray): The input grayscale image (NumPy array).

    Returns:
        np.ndarray: The contrast-enhanced grayscale image.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image_gray)


def detect_and_crop_hand_cv(
        image_gray: np.ndarray,
        offset_percent: int = OFFSET_PERCENT,
        blur_ksize: int = BLUR_KERNEL_SIZE,
        min_area_factor: float = MIN_CONTOUR_AREA_FACTOR) -> Optional[np.ndarray]:
    """
    Detects the main hand region in a grayscale X-ray image and crops it.

    This function uses OpenCV's contour detection to find the largest connected
    component (assumed to be the hand) and crops the image around its bounding box,
    applying an optional offset.

    Args:
        image_gray (np.ndarray): The input grayscale image.
        offset_percent (int, optional): Percentage to expand the bounding box. Defaults to OFFSET_PERCENT.
        blur_ksize (int, optional): Kernel size for initial Gaussian blur. Defaults to BLUR_KERNEL_SIZE.
        min_area_factor (float, optional): Minimum contour area relative to image area. Defaults to MIN_CONTOUR_AREA_FACTOR.

    Returns:
        Optional[np.ndarray]: The cropped image containing the hand, or None if
                              no suitable hand contour is found or cropping fails.
    """
    h, w = image_gray.shape

    # Apply Gaussian blur to reduce noise and smooth the image, aiding contour detection.
    blurred = cv2.GaussianBlur(image_gray, (blur_ksize, blur_ksize), 0)

    # Apply Otsu's thresholding to convert the grayscale image to a binary image.
    # This automatically determines an optimal threshold value.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours in the binary image. RETR_EXTERNAL retrieves only the outer contours.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return None as hand detection failed.
    if not contours:
        return None

    # Find the largest contour by area, assuming it corresponds to the hand.
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    min_area = h * w * min_area_factor

    # If the largest contour's area is too small, it's likely noise, so return None.
    if area < min_area:
        return None

    # Get the bounding box coordinates (x, y, width, height) of the largest contour.
    x, y, wb, hb = cv2.boundingRect(largest_contour)

    # Calculate the offset for the bounding box based on the specified percentage,
    # ensuring the cropped region includes a margin around the hand.
    offset_h = int((h * offset_percent) / 100)
    offset_w = int((w * offset_percent) / 100)

    # Calculate the final coordinates for cropping, ensuring they stay within image boundaries.
    y_min = max(0, y - offset_h)
    y_max = min(h, y + hb + offset_h)
    x_min = max(0, x - offset_w)
    x_max = min(w, x + wb + offset_w)

    # Crop the original grayscale image using the calculated bounding box.
    # Ensure the cropping dimensions are valid (max > min).
    if y_max > y_min and x_max > x_min:
        return image_gray[y_min:y_max, x_min:x_max]
    else:  # If calculated box is invalid (e.g., negative or zero dimensions)
        return None


def resize(image_gray: np.ndarray, output_size: int = OUTPUT_SIZE) -> np.ndarray:
    """
    Resizes a grayscale image to a square output_size, preserving aspect ratio by padding.

    The image is first padded to a square shape based on its maximum dimension,
    using the mean pixel value as padding color. Then, it's resized to the final
    desired output size.

    Args:
        image_gray (np.ndarray): The input grayscale image.
        output_size (int, optional): The target square dimension (e.g., 500x500). Defaults to OUTPUT_SIZE.

    Returns:
        np.ndarray: The resized and padded grayscale image.
    """
    h, w = image_gray.shape
    max_dim = max(h, w)

    # Calculate padding needed to make the image square.
    top_pad = (max_dim - h) // 2
    bottom_pad = max_dim - h - top_pad
    left_pad = (max_dim - w) // 2
    right_pad = max_dim - w - left_pad

    # Pad the image with the mean pixel value of the original image.
    padding = int(np.mean(image_gray))
    padded_img = cv2.copyMakeBorder(image_gray, top_pad, bottom_pad, left_pad, right_pad,
                                    cv2.BORDER_CONSTANT, value=padding)

    # Resize the padded image to the final output size using area interpolation (good for shrinking).
    resized_img = cv2.resize(padded_img, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return resized_img


def aug(image: np.ndarray) -> np.ndarray:
    """
    Applies a set of random augmentations to a grayscale image.

    These augmentations are typically used during training to increase the diversity
    of the dataset and improve the model's generalization capabilities.
    Applied augmentations include: random horizontal flip, random 90/180/270 degree rotations,
    random inversion (negative image), random brightness adjustment, and random contrast adjustment.

    Args:
        image (np.ndarray): The input grayscale image (NumPy array).

    Returns:
        np.ndarray: The augmented image.
    """
    # Random horizontal flip: Flips the image left-right with 50% probability.
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # Random 90/180/270 degree rotation: Rotates the image with 75% probability.
    # None means no rotation.
    rot_code = random.choice([None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
    if rot_code is not None:
        image = cv2.rotate(image, rot_code)

    # Random inversion (negative image): Inverts pixel values (255-x) with 50% probability.
    if random.random() > 0.5:
        image = 255 - image  # Invert pixel values for 8-bit grayscale (0-255 range)

    # Random brightness adjustment: Adjusts brightness by a random factor (0.7 to 1.3) with 50% probability.
    # `cv2.convertScaleAbs` handles clipping to 0-255 and converts to absolute values.
    if random.random() > 0.5:
        alpha = random.uniform(0.7, 1.3)  # Factor for brightness (1.0 is no change)
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)  # beta is offset, set to 0

    # Random contrast adjustment: Adjusts contrast by a random factor (0.7 to 1.3) with 50% probability.
    if random.random() > 0.5:
        alpha = random.uniform(0.7, 1.3)  # Factor for contrast
        # `cv2.convertScaleAbs` (alpha, beta) performs: output = alpha * input + beta
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)

    return image


def calculate_mean_std(image_folder: str, img_size: Tuple[int, int]) -> Tuple[float, float]:
    """
    Calculates the mean and standard deviation of pixel values across all images
    within a specified folder.

    This function iterates through all PNG images in the given directory,
    resizes them (if necessary), flattens their pixel values, and then computes
    the overall mean and standard deviation. It includes detailed logging for
    files that cannot be read or processed. These calculated values are crucial
    for standardizing the dataset for neural network input.

    Args:
        image_folder (str): The path to the directory containing the images (e.g., 'data/Train/prep_images').
        img_size (Tuple[int, int]): The target size (height, width) to which images should be
                                     resized before calculating pixel values. This ensures
                                     consistency in the calculation.

    Returns:
        Tuple[float, float]: A tuple containing the mean and standard deviation of all
                             processed pixel values. Returns (0.0, 1.0) if no valid
                             pixel values are found to prevent division by zero errors.
    """
    pixel_values = []
    print(f"\nCalculating mean and standard deviation for images in: {image_folder}")
    # List all PNG files in the specified folder.
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]

    total_files = len(image_files)
    processed_count = 0
    skipped_count = 0

    # Iterate through each image file with a progress bar.
    for img_file in tqdm(image_files, desc="Processing images for mean/std"):
        img_path = os.path.join(image_folder, img_file)
        try:
            # Preliminary check for empty or corrupted files by checking file size.
            if os.path.getsize(img_path) == 0:
                print(f"Warning: Skipped empty file {img_file}")
                skipped_count += 1
                continue

            # Read the image as grayscale directly using OpenCV.
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                # cv2.imread returns None if it fails to read the file (e.g., corrupted, wrong format).
                print(f"Warning: Could not read or process {img_file}. Skipping.")
                skipped_count += 1
                continue

            # Resize the image to the target size if its dimensions don't match.
            # This ensures all images contribute equally to the mean/std calculation,
            # regardless of their initial cropped size variability.
            if img.shape[0] != img_size[0] or img.shape[1] != img_size[1]:
                img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

            # Flatten the 2D image array into a 1D array of pixel values and extend the list.
            pixel_values.extend(img.flatten())
            processed_count += 1
        except Exception as e:
            # Catch any unexpected errors during image processing and log a warning.
            print(f"Warning: An unexpected error occurred while processing {img_file}: {e}. Skipping.")
            skipped_count += 1

    # Convert the list of pixel values to a NumPy array for efficient calculation.
    pixel_values = np.array(pixel_values, dtype=np.float32)

    # Print a summary of the mean/std calculation process.
    print("-" * 50)
    print(f"Mean/Std Calculation Summary for {image_folder}:")
    print(f"Total files found: {total_files}")
    print(f"Successfully processed files: {processed_count}")
    print(f"Skipped files: {skipped_count}")
    print("-" * 50)

    # If no valid pixel values were found (e.g., all files skipped), return default
    # values to prevent division by zero errors in standardization.
    if len(pixel_values) == 0:
        print("No valid pixel values found for mean/std calculation. Returning default values (0.0, 1.0).")
        return 0.0, 1.0

    # Calculate the mean and standard deviation of the collected pixel values.
    mean_val = np.mean(pixel_values)
    std_val = np.std(pixel_values)

    # Safeguard against zero standard deviation (e.g., if all pixels have the same value).
    # In such cases, replace std_val with 1.0 to prevent division by zero during standardization.
    if std_val < 1e-7:  # Using a small epsilon to check for near-zero std
        std_val = 1.0

    print(f"Mean pixel value of the training set: {mean_val:.4f}")
    print(f"Standard deviation of the training set pixel values: {std_val:.4f}")
    return mean_val, std_val