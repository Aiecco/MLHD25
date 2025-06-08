import os
import sys
import time

import cv2
from tqdm import tqdm

from src.preprocessing.PreprocessImage import detect_and_crop_hand_cv, apply_clahe, aug, resize

valid_sample_dir = 'validation_samples'
test_sample_dir = 'test_samples'
train_sample_dir = 'train_samples'
prep_img_dir = 'prep_images'

def preprocess_pipeline(folder, Train=False, Test=False, Val=True):
    if Val:
        subdir_input = os.path.join(folder, valid_sample_dir)
        subdir_output = os.path.join(folder, prep_img_dir)
        preprocess_images(subdir_input, subdir_output)
    elif Train:
        subdir_input = os.path.join(folder, train_sample_dir)
        subdir_output = os.path.join(folder, prep_img_dir)
        preprocess_images(subdir_input, subdir_output, augment=True)
    else:
        subdir_input = os.path.join(folder, test_sample_dir)
        subdir_output = os.path.join(folder, prep_img_dir)
        preprocess_images(subdir_input, subdir_output)

def preprocess_images(input_path, output_path, output_size=128, augment=False):

    print(f"Input directory:  {input_path}")
    print(f"Output directory: {output_path}")
    print(f"Output size:      {output_size}x{output_size} pixels")
    print(f"Augmentation:     {'Disabled' if not augment else 'Enabled'}")
    print("-" * 30)

    image_files = {img for img in os.listdir(input_path)}
    if not image_files:
        print(f"No PNG images found in {input_path}.")
        sys.exit(0)

    total_images = len(image_files)
    processed_count = 0
    success_count = 0
    skipped_count = 0
    start_time = time.time()

    for img_path in tqdm(image_files, desc="Processing Images", unit="img"):   # tqdm progress bar
        input_img_path = os.path.join(input_path, img_path)
        try:
            # read image as grayscale directly
            img_gray = cv2.imread(str(input_img_path), cv2.IMREAD_GRAYSCALE)
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
            img_resized = resize(img_clahe, output_size=output_size)

            #Augment (only train)
            img_final = img_resized
            if augment:
                img_final = aug(img_resized)

            # saver
            output_filename = os.path.join(output_path, img_path)
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
