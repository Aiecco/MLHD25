import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def plot_tensor_image(tensor, title="Processed Image"):
    """
    Plotta un'immagine a partire da un tensore TensorFlow.

    :param tensor: Tensore TensorFlow con shape (1, H, W) o (H, W).
    :param title: Titolo del plot.
    """
    if isinstance(tensor, tf.Tensor):
        tensor = tensor.numpy()  # Converte in NumPy array

    # Se il tensore ha la dimensione (1, H, W), rimuove il canale
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]

    plt.figure(figsize=(5, 5))
    plt.imshow(tensor, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

def preprocess_pooled_and_heatmap(image_path, img_size=(128, 128), normalize=True):
    """
    Preprocessa un'immagine generando due rappresentazioni avanzate:
      - 'Pooled': immagine ridimensionata, con CLAHE applicato e normalizzata.
      - 'Heatmap': mappa avanzata dei gradienti ottenuta con:
                   - filtro Sobel multi-scala
                   - rilevamento dei bordi Canny
                   - fusione di caratteristiche

    :param image_path: Percorso al file immagine.
    :param img_size: Dimensione target (larghezza, altezza) dell'immagine.
    :param normalize: Se True, normalizza:
                      - per pooled: valori in [-1, 1]
                      - per heatmap: valori in [0, 1]
    :return: (pooled_tensor, heatmap_tensor) tensori TensorFlow con shape (1, H, W)
    """
    # Legge l'immagine in scala di grigi
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Impossibile leggere l'immagine: {image_path}")

    # Ridimensiona l'immagine
    image_resized = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)

    # ---- Pooled Branch (Enhanced with CLAHE) ----
    # Applica CLAHE (Contrast Limited Adaptive Histogram Equalization) per migliorare il contrasto locale
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(image_resized)
    
    # Converti in float32 per ulteriori operazioni
    pooled_img = clahe_img.astype(np.float32)
    
    if normalize:
        # Normalizza i pixel in [-1, 1]
        pooled_img = (pooled_img - 127.5) / 127.5
    
    # Aggiungi dimensione del canale: (1, H, W)
    pooled_img = np.expand_dims(pooled_img, axis=0)
    pooled_tensor = tf.convert_to_tensor(pooled_img, dtype=tf.float32)

    # ---- Heatmap Branch (Enhanced with multi-scale gradients and Canny) ----
    
    # 1. Multi-scale Sobel gradients (3 scale levels)
    grad_magnitudes = []
    
    # Scale 1: kernel size 3 (dettagli più fini)
    sobelx_3 = cv2.Sobel(image_resized, cv2.CV_64F, 1, 0, ksize=3)
    sobely_3 = cv2.Sobel(image_resized, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag_3 = np.sqrt(sobelx_3 ** 2 + sobely_3 ** 2)
    grad_magnitudes.append(grad_mag_3)
    
    # Scale 2: kernel size 5 (dettagli medi)
    sobelx_5 = cv2.Sobel(image_resized, cv2.CV_64F, 1, 0, ksize=5)
    sobely_5 = cv2.Sobel(image_resized, cv2.CV_64F, 0, 1, ksize=5)
    grad_mag_5 = np.sqrt(sobelx_5 ** 2 + sobely_5 ** 2)
    grad_magnitudes.append(grad_mag_5)
    
    # Scale 3: kernel size 7 (dettagli più ampi)
    sobelx_7 = cv2.Sobel(image_resized, cv2.CV_64F, 1, 0, ksize=7)
    sobely_7 = cv2.Sobel(image_resized, cv2.CV_64F, 0, 1, ksize=7)
    grad_mag_7 = np.sqrt(sobelx_7 ** 2 + sobely_7 ** 2)
    grad_magnitudes.append(grad_mag_7)
    
    # Combina i gradienti a diverse scale con pesi
    weights = [0.5, 0.3, 0.2]  # Enfatizza dettagli fini ma include anche scala più ampia
    multi_scale_grad = np.zeros_like(grad_mag_3)
    for i, grad in enumerate(grad_magnitudes):
        multi_scale_grad += weights[i] * grad
    
    # 2. Canny edge detection per complementare i gradienti Sobel
    # Usa valori adattivi per le soglie basati sull'istogramma dell'immagine
    median_val = np.median(image_resized)
    lower_threshold = int(max(0, (1.0 - 0.33) * median_val))
    upper_threshold = int(min(255, (1.0 + 0.33) * median_val))
    canny_edges = cv2.Canny(image_resized, lower_threshold, upper_threshold)
    canny_edges = canny_edges.astype(np.float64)
    
    # 3. Combina multi-scale gradient con Canny edges (70% gradient, 30% Canny)
    combined_features = 0.7 * multi_scale_grad + 0.3 * canny_edges
    
    if normalize:
        # Normalizza il risultato combinato in [0, 1]
        combined_features = combined_features / (combined_features.max() + 1e-8)
    else:
        combined_features = combined_features.astype(np.float32)
    
    # Aggiungi dimensione del canale: (1, H, W)
    heatmap_img = np.expand_dims(combined_features, axis=0)
    heatmap_tensor = tf.convert_to_tensor(heatmap_img, dtype=tf.float32)

    return pooled_tensor, heatmap_tensor