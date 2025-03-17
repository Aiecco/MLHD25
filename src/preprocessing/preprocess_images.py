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
    Preprocessa un'immagine generando due rappresentazioni:
      - 'Pooled': immagine ridimensionata e normalizzata.
      - 'Heatmap': mappa dei gradienti ottenuta con il filtro Sobel.

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

    # ---- Pooled Branch ----
    pooled_img = image_resized.astype(np.float32)
    if normalize:
        # Normalizza i pixel in [-1, 1]
        pooled_img = (pooled_img - 127.5) / 127.5
    # Aggiungi dimensione del canale: (1, H, W)
    pooled_img = np.expand_dims(pooled_img, axis=0)
    pooled_tensor = tf.convert_to_tensor(pooled_img, dtype=tf.float32)

    # ---- Heatmap Branch ----
    # Calcola i gradienti con l'operatore Sobel
    sobelx = cv2.Sobel(image_resized, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image_resized, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

    if normalize:
        # Normalizza il gradiente in [0, 1]
        grad_magnitude = grad_magnitude / (grad_magnitude.max() + 1e-8)
    else:
        grad_magnitude = grad_magnitude.astype(np.float32)

    # Aggiungi dimensione del canale: (1, H, W)
    heatmap_img = np.expand_dims(grad_magnitude, axis=0)
    heatmap_tensor = tf.convert_to_tensor(heatmap_img, dtype=tf.float32)

    return pooled_tensor, heatmap_tensor