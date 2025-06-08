import os.path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#from src.preprocessing.PreprocessImage import preprocess_hand_image
from src.utils.HandDetector import isolate_radiograph_area
from src.utils.ObjectDetector import use_decompose_image_display_all


def display_raw_prep(path, prep_out_path, extr_out_path, plot=False):
    """
    Legge un'immagine da `path`, la converte in grayscale [0,1],
    la ridimensiona a img_size, e poi plotta:
     - Raw
     - Laplacian in frequenza
     - HFE in frequenza
    """
    # 1) Leggi e decodifica
    name = path.split('\\')[-1]
    print(name)

    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)      # shape=(H, W, 1), uint8

    raw_np = img.numpy()

    prep_img = preprocess_hand_image(raw_np)

    # Salvo prep_img in out_path
    encoded_image = tf.io.encode_png(prep_img)
    file_path = os.path.join(prep_out_path, name)
    tf.io.write_file(file_path, encoded_image)
    print(f"Immagine preprocessata salvata in: {file_path}")

    extr_img_float = use_decompose_image_display_all(path)
    if extr_img_float is not None:
        # Converti l'immagine float32 nell'intervallo [0, 255] e poi in uint8
        extr_img = (extr_img_float * 255).astype(np.uint8)
        extr_img = tf.expand_dims(extr_img, axis=-1)

    # 5) Plot
    if plot:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, im, title in zip(axes,
                                 [raw_np, prep_img, extr_img],
                                 ['Raw', 'Preprocessed']):
            ax.imshow(im, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
        plt.tight_layout()
        plt.show()

    encoded_image = tf.io.encode_png(extr_img)
    file_path = os.path.join(extr_out_path, name)
    tf.io.write_file(file_path, encoded_image)
    print(f"Immagine preprocessata salvata in: {file_path}")
