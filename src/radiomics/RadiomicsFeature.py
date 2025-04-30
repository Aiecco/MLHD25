import os

import numpy as np
from skimage.feature import graycomatrix, graycoprops
import tensorflow as tf


# ----------------------------
# 2) Estrazione radiomica semplificata
# ----------------------------

# Utilità: estrazione radiomics tramite tf.py_function
def _extract_radiomics_py(image_path, filename_base, output_dir):
    """
    Funzione numpy per estrazione Haralick via skimage e salvataggio features.
    """
    try:
        image_path = image_path.numpy().decode('utf-8')
        # Carica l'immagine utilizzando TensorFlow (assicurati che sia un formato supportato)
        image_tensor = tf.io.read_file(image_path)
        image = tf.image.decode_image(image_tensor, channels=1).numpy()  # Forza in scala di grigi

        img_uint8 = (image.squeeze() * 255).astype(np.uint8)
        glcm = graycomatrix(img_uint8, distances=[1], angles=[0], levels=256,
                            symmetric=True, normed=True)
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy']
        feats = np.array([graycoprops(glcm, prop)[0, 0] for prop in props], dtype=np.float32)

        # Salva le features
        save_features(feats, filename_base, output_dir)

        return feats
    except Exception as e:
        print(f"Errore nell'elaborazione di {image_path}: {e}")
        return np.array([np.nan] * 4, dtype=np.float32)  # Restituisci un array di NaN in caso di errore


def extract_radiomics_tf(image_path, filename_base, output_dir):
    """
    Wrapper tf.py_function per estrazione radiomics.
    """
    feats = tf.py_function(
        _extract_radiomics_py,
        [image_path, filename_base, output_dir],
        tf.float32
    )
    feats.set_shape((4,))
    return feats


# Funzione per salvare le features in un file .npy
def save_features(features, filename, output_dir):
    """
    Salva le features numpy in un file nella directory specificata.
    """
    output_dir = output_dir.numpy().decode('utf-8')
    filename = filename.numpy().decode('utf-8')
    filepath = os.path.join(output_dir, f"{filename}.npy")
    np.save(filepath, features)
    print(f"Features salvate in: {filepath}")


def load_features(features_path):
    """
    Carica le features da un file .npy.
    """
    if isinstance(features_path, np.ndarray):
        features_path = features_path.item()
    if isinstance(features_path, bytes):
        features_path = features_path.decode('utf-8')

    try:
        features = np.load(features_path)
        return features
    except FileNotFoundError:
        print(f"File di features non trovato: {features_path}")
        return None
