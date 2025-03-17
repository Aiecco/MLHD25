import os

from src.preprocessing.preprocess_images import preprocess_pooled_and_heatmap


def preprocess_dataset(img_path):

    processed_img, heated_img = preprocess_pooled_and_heatmap(img_path)

    return processed_img, heated_img