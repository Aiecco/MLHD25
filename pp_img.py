import random
import time
import sys
import cv2
import os
import platform


def CLAHE(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE for improving image contrast
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return result


def AUGMENT(img):
    """
    Randomly flip or rotate the image to augment data.
    """
    if (random.random() > 0.5): img = cv2.flip(img, 1)  # Flip horizontally
    if (random.random() > 0.5): img = cv2.flip(img, -1)  # Flip vertically
    rnd = random.random()
    if rnd < 0.25: img = cv2.rotate(img, cv2.ROTATE_180)  # Rotate 180 degrees
    elif rnd < 0.5: img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)  # Rotate 90 degrees counterclockwise
    elif rnd < 0.75: img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # Rotate 90 degrees clockwise
    return img


def _print():
    now = time.time() - start_time
    print("%c[%d;%df" % (0x1B, 0, 0), end='')
    print(
        "PREPROCESSING..."
        "\nInput directory:  ", image_dir, "\nOutput directory: ", output_dir,
        "\nOutput Size: ", output_size,
        ("px | 3" if output_channel == 3 else "px | 1"), "Output channels ",
        "\nRunning time: %02d:%05.02f" % (int(now / 60), now % 60),
        f"\nProcessed images: {count}/{img_count} ({int(count / img_count * 100)}%)"
    )



# Initialize variables
count = 0  # Count of processed images

# Input and output directories
image_dir = os.path.abspath(str(sys.argv[1][:-1]))  # Path to the image directory
if (str(platform.system()).lower() == "windows"):
    output_dir = os.path.abspath(os.path.join(image_dir, os.pardir)) + "\\" + os.path.basename(
        image_dir) + "_pp\\"  # Output directory for Windows
else:
    output_dir = os.path.abspath(os.path.join(image_dir, os.pardir)) + "/" + os.path.basename(
        image_dir) + "_pp/"  # Output directory for Unix-based systems

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Count the number of images in the input directory
img_count = len([entry for entry in os.listdir(image_dir)
                 if os.path.isfile(os.path.join(image_dir, entry))])

output_size = 256  # Output image size
output_channel = 1  # Output image channels (grayscale)


start_time = time.time()

# Process each image in the input directory
for filename in os.listdir(image_dir):
    count += 1
    img = cv2.imread(os.path.join(image_dir, filename))
    if (img is None):
        continue
    else:
        img = CLAHE(img)  # Apply CLAHE to improve contrast
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)  # Resize the image to 256x256
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        # img = AUGMENT(img)  # ONLY for training set? - comment when preprocessing val and train
        cv2.imwrite(os.path.join(output_dir, filename), img)  # Save the processed image
        _print()  # Print progress
