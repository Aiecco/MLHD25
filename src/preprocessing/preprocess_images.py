import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms


def preprocess_image(image_path, img_size=(128, 128), normalize=True):
    """
    Preprocessa un'immagine per essere usata nel modello.

    :param image_path: Percorso del file immagine.
    :param img_size: Dimensione (altezza, larghezza) alla quale ridimensionare l'immagine.
    :param normalize: Se True, normalizza l'immagine tra -1 e 1.
    :return: Tensore PyTorch pre-elaborato, pronto per essere dato in input al modello.
    """

    # 1. Leggere l'immagine in scala di grigi
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Impossibile leggere l'immagine: {image_path}")

    # 2. Ridimensionare l'immagine
    image = cv2.resize(image, img_size, interpolation=cv2.INTER_AREA)

    # 3. Convertire l'immagine in un array numpy di tipo float32
    image = image.astype(np.float32)

    # 4. Normalizzazione (opzionale)
    if normalize:
        image = (image - 127.5) / 127.5  # Porta i valori tra -1 e 1

    # 5. Aggiungere la dimensione del canale (C, H, W) e convertire in tensore
    image = np.expand_dims(image, axis=0)  # Aggiunge il canale: (1, H, W)
    image_tensor = torch.tensor(image, dtype=torch.float32)

    return image_tensor


def plot_tensor_image(tensor, title="Processed Image"):
    """
    Plotta un'immagine a partire da un tensore PyTorch.

    :param tensor: Tensore PyTorch con shape (1, H, W) o (H, W).
    :param title: Titolo del plot.
    """
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cpu().detach().numpy()  # Converte in NumPy array

    # Se il tensore ha la dimensione (1, H, W), rimuove il canale
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]

    plt.figure(figsize=(5, 5))
    plt.imshow(tensor, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()