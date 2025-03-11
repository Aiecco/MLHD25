import os

from src.preprocessing.preprocess_images import preprocess_image


def preprocess(img_path):
    processed_imgs = []
    files = {f for f in os.listdir(img_path)}
    for f in files:
        processed_img = preprocess_image(img_path)
        print("Processed Image Shape:", processed_imgs.shape)

        processed_imgs.append(processed_img)

    return processed_imgs
