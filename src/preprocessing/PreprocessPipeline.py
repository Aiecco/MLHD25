import os
import sys
import time
import cv2
from tqdm import tqdm

# Import specific preprocessing functions from PreprocessImage module
from src.preprocessing.PreprocessImage import detect_and_crop_hand_cv, apply_clahe, aug, resize

# --- Constants for Directory Names ---
valid_sample_dir = 'validation_samples'
"""str: Subdirectory name for validation raw images."""
test_sample_dir = 'test_samples'
"""str: Subdirectory name for test raw images."""
train_sample_dir = 'train_samples'
"""str: Subdirectory name for training raw images."""
prep_img_dir = 'prep_images'
"""str: Subdirectory name where preprocessed images will be saved."""


def preprocess_pipeline(folder: str, Train: bool = False, Test: bool = False, Val: bool = True):
    """
    Orchestrates the preprocessing of image datasets based on the specified set type.

    This function determines which input subdirectory to use (train, validation, or test)
    and calls the `preprocess_images` function accordingly, enabling augmentation
    only for the training set.

    Args:
        folder (str): The base directory containing the raw image subfolders (e.g., 'data/Train').
        Train (bool, optional): If True, preprocesses the training samples and applies augmentation. Defaults to False.
        Test (bool, optional): If True, preprocesses the test samples (no augmentation). Defaults to False.
        Val (bool, optional): If True, preprocesses the validation samples (no augmentation). Defaults to True.
                              Note: Only one of Train, Test, or Val should typically be True.
    """
    if Val:  # Preprocess validation samples
        subdir_input = os.path.join(folder, valid_sample_dir)
        subdir_output = os.path.join(folder, prep_img_dir)
        preprocess_images(subdir_input, subdir_output, augment=False)  # No augmentation for validation
    elif Train:  # Preprocess training samples
        subdir_input = os.path.join(folder, train_sample_dir)
        subdir_output = os.path.join(folder, prep_img_dir)
        preprocess_images(subdir_input, subdir_output, augment=True)  # Apply augmentation for training
    elif Test:  # Preprocess test samples
        subdir_input = os.path.join(folder, test_sample_dir)
        subdir_output = os.path.join(folder, prep_img_dir)
        preprocess_images(subdir_input, subdir_output, augment=False)  # No augmentation for test
    else:
        print("Warning: No dataset type (Train, Test, Val) specified for preprocessing. No action taken.")


def preprocess_images(input_path: str, output_path: str, output_size: int = 256, augment: bool = False):
    """
    Processes images from an input directory and saves the processed versions
    to an output directory.

    Each image undergoes hand detection, cropping, CLAHE enhancement, resizing,
    and optional data augmentation. Provides progress updates and error logging.

    Args:
        input_path (str): The directory containing the raw input images.
        output_path (str): The directory where processed images will be saved.
                           This directory will be created if it does not exist.
        output_size (int, optional): The target side length for resized square images. Defaults to 256.
        augment (bool, optional): If True, applies random data augmentation to images
                                  before saving. Typically True for training sets only. Defaults to False.
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    print(f"Input directory:  {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Output size:      {output_size}x{output_size} pixels")
    print(f"Augmentation:     {'Disabled' if not augment else 'Enabled'}")
    print("-" * 30)

    # List all PNG files in the input directory.
    # Using a set initially for potential future deduplication if needed, though list comprehension is often fine.
    image_files = {img for img in os.listdir(input_path) if img.lower().endswith('.png')}

    if not image_files:
        print(f"No PNG images found in {input_path}. Exiting preprocessing for this folder.")
        # sys.exit(0) # Do not exit the whole script if one folder is empty.
        return  # Just return from the function

    total_images = len(image_files)
    processed_count = 0  # Count of images successfully processed and saved
    success_count = 0  # Count of images where hand detection was successful
    skipped_count = 0  # Count of images skipped due to read errors or detection failure
    start_time = time.time()

    # Iterate through each image file with a progress bar.
    for img_file in tqdm(image_files, desc="Processing Images", unit="img"):
        input_img_path = os.path.join(input_path, img_file)
        try:
            # Read the image as grayscale. cv2.IMREAD_GRAYSCALE ensures 1 channel.
            img_gray = cv2.imread(str(input_img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:  # Check if image reading failed (e.g., corrupted file)
                print(f"Warning: Could not read image {img_file}. Skipping.")
                skipped_count += 1
                continue

            # Detect and crop the hand region from the grayscale image.
            img_cropped = detect_and_crop_hand_cv(img_gray)

            if img_cropped is None:  # Check if hand detection failed for this image
                print(f"Warning: Hand detection failed for {img_file}. Skipping.")
                skipped_count += 1
                continue
            else:
                success_count += 1  # Increment count for successful hand detections

            # Apply CLAHE to the cropped image to enhance contrast.
            img_clahe = apply_clahe(img_cropped)

            # Resize the image to the target output_size, preserving aspect ratio with padding.
            img_resized = resize(img_clahe, output_size=output_size)

            # Apply data augmentation if `augment` flag is True (typically for training set).
            img_final = img_resized
            if augment:
                img_final = aug(img_resized)

            # Construct the output filename and save the final processed image.
            output_filename = os.path.join(output_path, img_file)
            cv2.imwrite(str(output_filename), img_final)
            processed_count += 1  # Increment count for successfully processed and saved images

        except Exception as e:
            # Catch any unexpected errors during the processing of a single image.
            print(f"Error processing {img_file}: {e}. Skipping.")
            skipped_count += 1
            continue  # Continue to the next image even if an error occurs

    # --- Summarize processing results ---
    elapsed_time = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print("\n" + "-" * 30)
    print("Image preprocessing finished.")
    print(f"Total images found in input directory: {total_images}")
    print(f"Successfully processed and saved: {processed_count}")
    print(
        f"Hand detection successful: {success_count}")  # Note: success_count refers to hand detection, not file saving.
    print(f"Images skipped (read/detection error): {skipped_count}")
    print(f"Total time elapsed: {elapsed_str}")
    if total_images > 0:
        # Calculate hand detection success rate based on files attempted to process.
        detection_rate = (success_count / (processed_count + skipped_count) * 100) if (
                                                                                                  processed_count + skipped_count) > 0 else 0
        print(f"Hand Detection Success Rate (of processed/skipped): {detection_rate:.1f}%")
    print("-" * 30)