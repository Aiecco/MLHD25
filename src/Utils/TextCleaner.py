import cv2
import numpy as np


def remove_text_by_intensity(image_path, background_color=(0), threshold_value=220, max_value=255, kernel_size=(5, 5)):
    """
    Rimuove le scritte dall'immagine basandosi sulla loro intensità, riempiendo con il colore di sfondo.

    Args:
        image_path (str): Percorso del file immagine.
        background_color (tuple): Colore (in scala di grigi) da usare per riempire le aree di testo.
                                  Per le radiografie scure, probabilmente 0 (nero).
        threshold_value (int): Valore di soglia per isolare le scritte.
        max_value (int): Valore massimo usato nella soglia.
        kernel_size (tuple): Dimensione del kernel per le operazioni morfologiche (chiusura).

    Returns:
        np.ndarray: L'immagine con le scritte rimosse come array NumPy.
    """
    # Carica l'immagine in scala di grigi usando OpenCV

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Errore nel caricare l'immagine: {image_path}")
        return None

    # Applica una soglia per isolare le scritte
    # Supponiamo che le scritte siano più chiare dello sfondo scuro.
    # Se le scritte sono più scure, usa cv2.THRESH_BINARY_INV
    _, thresh = cv2.threshold(img, threshold_value, max_value, cv2.THRESH_BINARY)

    # Applica operazioni morfologiche per migliorare la maschera delle scritte
    # Un'operazione di chiusura (dilatazione seguita da erosione) può aiutare a unire
    # parti separate delle lettere e a rimuovere piccoli "buchi".
    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Inverti la maschera: vogliamo mantenere tutto tranne le scritte
    mask_inv = cv2.bitwise_not(mask)

    # Crea un'immagine di sfondo dello stesso colore dell'area da riempire
    background = np.full(img.shape, background_color, dtype=np.uint8)

    # Sovrapponi l'immagine originale (dove la maschera_inv è bianca) e lo sfondo (dove la maschera è bianca)
    # Questo riempie le aree della maschera (le scritte) con il colore di sfondo.
    img_cleaned = cv2.bitwise_and(img, img, mask=mask_inv)
    background_area = cv2.bitwise_and(background, background, mask=mask)
    img_cleaned = cv2.add(img_cleaned, background_area)


    return img_cleaned