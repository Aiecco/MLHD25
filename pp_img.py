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
    # random horizontal flip
    if random.random() > 0.5:
        image = cv2.flip(image, 1)

    # random 90/180/270 deg rotation
    rot_code = random.choice([None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
    if rot_code is not None:
        image = cv2.rotate(image, rot_code)
    return image



# -- Execution pipeline --

def main():
    parser = argparse.ArgumentParser(description="Preprocess hand X-ray images for bone age prediction.")
    parser.add_argument("input_dir", type=str, help="Directory containing input PNG images.")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="Output directory for processed images. Defaults to '<input_dir>_pp'.")
    parser.add_argument("-s", "--size", type=int, default=OUTPUT_SIZE,
                        help=f"Output image size (default: {OUTPUT_SIZE}x{OUTPUT_SIZE}).")
    parser.add_argument("--no_augment", action="store_true",
                        help="Disable image augmentation (flips/rotations).")
    args = parser.parse_args()

    input_path = Path(args.input_dir).resolve()
    if not input_path.is_dir():
        print(f"Error: Input directory not found: {input_path}")
        sys.exit(1)

    output_path = Path(args.output_dir) if args.output_dir else input_path.parent / f"{input_path.name}_pp"
    output_path.mkdir(parents=True, exist_ok=True)


    print(f"Input directory:  {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Output size:      {OUTPUT_SIZE}x{OUTPUT_SIZE} pixels")
    print(f"Augmentation:     {'Disabled' if args.no_augment else 'Enabled'}")
    print("-" * 30)

    image_files = list(input_path.glob('*.png')) # all are PNGs
    if not image_files:
        print(f"No PNG images found in {input_path}.")
        sys.exit(0)

    total_images = len(image_files)
    processed_count = 0
    success_count = 0
    skipped_count = 0
    start_time = time.time()

    for img_path in tqdm(image_files, desc="Processing Images", unit="img"):   # tqdm progress bar
        try:
            # read image as grayscale directly
            img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                skipped_count += 1 # skip on error
                continue

            # detect + crop hand
            img_cropped = detect_and_crop_hand_cv(img_gray)

            if img_cropped is None:
                skipped_count += 1 # skip on error
                continue
            else:
                success_count += 1 # Count successful detections

            # apply clahe (on cropped grayscale)
            img_clahe = apply_clahe(img_cropped)

            # resize w/ Padding
            img_resized = resize(img_clahe, output_size=OUTPUT_SIZE)

            #Augment (only train)
            img_final = img_resized
            if not args.no_augment:
                img_final = aug(img_resized)

            # saver
            output_filename = output_path / img_path.name
            cv2.imwrite(str(output_filename), img_final)
            processed_count += 1 # successfully processed imgs

        except Exception as e:
            skipped_count += 1
            continue # skip on error

    # summarizing descr
    elapsed_time = time.time() - start_time
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print("\n" + "-" * 30)
    print("Processing finished.")
    print(f"Total images found: {total_images}")
    print(f"Successfully processed: {processed_count}")
    print(f"Hand detection successful: {success_count}")
    print(f"Images skipped (read/detection error): {skipped_count}")
    print(f"Total time: {elapsed_str}")
    if total_images > 0:
      detection_rate = (success_count / (processed_count + skipped_count) * 100) if (processed_count + skipped_count) > 0 else 0
      print(f"Hand Detection Success Rate: {detection_rate:.1f}%")


if __name__ == "__main__":
    main()