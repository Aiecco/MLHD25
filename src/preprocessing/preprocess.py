import os

from src.preprocessing.preprocess_images import preprocess_image


def preprocess_dataset(img_path):

    processed_img = preprocess_image(img_path)

    return processed_img
