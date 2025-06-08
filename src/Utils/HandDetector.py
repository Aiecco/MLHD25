import cv2
import numpy as np
import matplotlib.pyplot as plt # Per visualizzare i risultati

def isolate_radiograph_area(image_path, padding=10):
    """
    Identifica e ritaglia l'area della radiografia da un'immagine con bordo scuro.

    Args:
        image_path (str): Percorso del file immagine.
        padding (int): Pixel di padding da aggiungere attorno all'area ritagliata.

    Returns:
        np.ndarray: L'area ritagliata della radiografia come array NumPy in scala di grigi,
                    o l'immagine originale se l'isolamento fallisce.
        tuple: Le coordinate del bounding box trovato (x, y, w, h) sull'immagine originale, o (0, 0, img_w, img_h).
    """
    # Carica l'immagine in scala di grigi
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Errore nel caricare l'immagine: {image_path}")
        return None, None

    img_h, img_w = img.shape

    # --- 1. Soglia per separare l'area luminosa ---
    # Applica una soglia per creare un'immagine binaria dove l'area della radiografia è bianca
    # e lo sfondo scuro è nero.
    # Usiamo OSTU per trovare la soglia automaticamente, assumendo un istogramma bimodale
    # (picco per lo sfondo scuro, picco per la radiografia più chiara).
    # Se OSTU non funziona bene, potresti dover usare una soglia fissa.
    try:
        # Applica una leggera sfocatura prima della soglia per ridurre il rumore
        img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
        thresh_value, thresh = cv2.threshold(img_blurred, 200, 100, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Se THRESH_BINARY + OSTU rende lo sfondo bianco e la radiografia nera, inverti
        # Controlla l'istogramma o un campione per capire se l'area più grande (sfondo o radiografia)
        # è stata resa bianca. Se il bianco è lo sfondo che vuoi rimuovere, inverti la soglia.
        # Un controllo euristico: se l'area bianca copre > 50% dell'immagine, potrebbe essere lo sfondo.
        if np.sum(thresh) / 255 > (img_h * img_w) / 2: # Se l'area bianca è più della metà...
             _, thresh = cv2.threshold(img_blurred, 200, 100, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


    except Exception as e:
        print(f"Errore durante la soglia OSTU per {image_path}: {e}. Riprovo con soglia fissa e inverti.")
        # Se OSTU fallisce, usa una soglia fissa e poi inverti (assumendo sfondo scuro)
        thresh_value = 50 # Valore fisso da regolare se necessario
        _, thresh = cv2.threshold(img, thresh_value, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_not(thresh) # Inverti per rendere la radiografia bianca


    # --- 2. Trova i contorni delle aree bianche ---
    # I contorni delimitano le regioni nell'immagine binaria.
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"Nessun contorno significativo trovato nell'immagine: {image_path}. Ritorno immagine originale.")
        return img.copy(), (0, 0, img_w, img_h) # Se non trovi contorni, ritorna l'immagine originale

    # --- 3. Identifica il contorno più grande (l'area della radiografia) ---
    # Assumiamo che l'area della radiografia sia il contorno più grande.
    largest_contour = max(contours, key=cv2.contourArea)

    # --- 4. Ottieni il bounding box del contorno più grande ---
    # Il bounding box è il rettangolo più piccolo che racchiude il contorno.
    x, y, w, h = cv2.boundingRect(largest_contour)

    # --- 5. Applica padding al bounding box e ritaglia l'immagine originale ---
    # Aggiungi un po' di spazio extra attorno al bounding box per sicurezza.
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(img_w, x + w + padding)
    y_end = min(img_h, y + h + padding)

    # Ritaglia l'immagine originale (in scala di grigi) usando le coordinate del bounding box allargato
    cropped_img = img[y_start:y_end, x_start:x_end]

    return cropped_img, (x_start, y_start, x_end - x_start, y_end - y_start)
