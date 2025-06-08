import cv2
import numpy as np
import matplotlib.pyplot as plt

def decompose_image_into_objects(img, threshold_method=cv2.THRESH_BINARY + cv2.THRESH_OTSU, threshold_value=100, morphological_kernel_size=(5, 5)):
    """
    Scompone un'immagine in scala di grigi in oggetti basati sulle componenti connesse
    dopo soglia e operazioni morfologiche.

    Args:
        image_path (str): Percorso del file immagine.
        threshold_method (int): Metodo di soglia di OpenCV (es. cv2.THRESH_BINARY, cv2.THRESH_BINARY + cv2.THRESH_OTSU).
        threshold_value (int): Valore di soglia (usato solo se threshold_method non include OSTU).
        morphological_kernel_size (tuple): Dimensione del kernel per le operazioni morfologiche.

    Returns:
        tuple: Un tuple contenente:
            - np.ndarray: L'immagine originale in scala di grigi.
            - list: Una lista di tuple, dove ogni tuple contiene (label, component_mask, stats, centroid)
                    per ogni componente connessa trovata (escludendo lo sfondo).
    """
    # Carica l'immagine in scala di grigi

    # --- 1. Applica una soglia ---
    # Applica una leggera sfocatura prima della soglia per ridurre il rumore
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)

    if threshold_method == cv2.THRESH_BINARY + cv2.THRESH_OTSU or threshold_method == cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU:
         try:
             thresh_value_auto, thresh = cv2.threshold(img_blurred, 0, 255, threshold_method)
             print(f"Soglia OSTU automatica: {thresh_value_auto}")
         except Exception as e:
             print(f"Errore durante soglia OSTU: {e}. Utilizzo soglia fissa.")
             _, thresh = cv2.threshold(img_blurred, threshold_value, 255, cv2.THRESH_BINARY) # O THRESH_BINARY_INV

    else:
        _, thresh = cv2.threshold(img_blurred, threshold_value, 255, threshold_method)


    # --- 2. Pulizia con operazioni morfologiche ---
    # Applica un'operazione di chiusura per riempire piccoli buchi e unire aree vicine.
    # Sperimenta con l'ordine (opening, closing) e la dimensione del kernel.
    kernel = np.ones(morphological_kernel_size, np.uint8)
    cleaned_thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Potresti voler aggiungere anche un'opening per rimuovere piccoli rumori
    # cleaned_thresh = cv2.morphologyEx(cleaned_thresh, cv2.MORPH_OPEN, kernel)


    # --- 3. Trova le componenti connesse ---
    # cv2.connectedComponentsWithStats restituisce:
    # - num_labels: il numero totale di componenti (incluso lo sfondo)
    # - labels: un'immagine dove ogni pixel ha un valore che indica a quale componente appartiene
    # - stats: statistiche per ogni componente (area, bounding box, ecc.)
    # - centroids: le coordinate del centroide per ogni componente
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned_thresh, 8, cv2.CV_32S)

    # --- 4. Analizza e prepara i risultati per ogni componente ---
    # La componente 0 è sempre lo sfondo, quindi partiamo da 1.
    components = []
    img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Converti in BGR per disegnare a colori

    for i in range(1, num_labels):
        # Ottieni le statistiche e il centroide per la componente i
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        # Crea una maschera per la singola componente
        component_mask = (labels == i).astype(np.uint8) * 255

        # Puoi filtrare le componenti qui in base all'area o ad altre statistiche
        # if area > min_area_threshold and area < max_area_threshold: # Esempio di filtro per area
        components.append((i, component_mask, stats[i], centroids[i]))

        # Esempio: Disegna il bounding box e il numero della componente sull'immagine originale
        cv2.rectangle(img_display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img_display, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return img, components

def use_decompose_image_display_all(image_file, plot=False):
    """
    Usa decompose_image_into_objects, visualizza ogni componente singolarmente,
    e infine visualizza l'immagine originale senza gli oggetti trovati.
    """
    # Descomponi l'immagine in oggetti
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Errore nel caricare l'immagine: {image_file}")
        return None, None

    img_original_gray, found_components = decompose_image_into_objects(
        img,
        threshold_method=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        morphological_kernel_size=(20, 20)
    )

    if img_original_gray is not None: # Continua anche se non ci sono componenti trovate oltre lo sfondo
        print(f"Trovate {len(found_components)} componenti connesse (escluso sfondo).")

        # --- Visualizza ogni componente trovata, una per volta (come prima) ---
        if found_components:
            print("Visualizzazione delle singole componenti...")
            for label, mask, stats, centroid in found_components:
                area = stats[cv2.CC_STAT_AREA]
                if plot:
                    plt.imshow(mask, cmap='gray')
                    plt.title(f"Componente {label} - Area: {area}")
                    plt.axis('off')
                    plt.show()
            print("Visualizzazione delle singole componenti completata.")
        else:
            print("Nessuna componente significativa trovata per la visualizzazione singola.")


        # --- Crea l'immagine senza gli oggetti trovati ---
        # Ottieni l'immagine 'labels' dalla chiamata originale a connectedComponentsWithStats
        # Dobbiamo passare questa informazione fuori da decompose_image_into_objects
        # Modificheremo decompose_image_into_objects per restituire anche l'immagine labels

        # PER ORA, assumiamo di poter ottenere l'immagine 'labels' in qualche modo qui
        # In una versione reale, dovresti modificare decompose_image_into_objects per restituirla
        # Esempio fittizio di come potresti ottenere 'labels':
        # _, labels, _, _ = cv2.connectedComponentsWithStats(...) # Questo dovrebbe venire dalla funzione

        # Alternativa: ricrea la maschera di tutti gli oggetti (escluso lo sfondo)
        # Partiamo da un'immagine completamente nera delle stesse dimensioni
        all_objects_mask = np.zeros_like(img_original_gray, dtype=np.uint8)
        # Disegna ogni componente trovata sulla maschera (come bianco)
        for label, mask, stats, centroid in found_components:
             all_objects_mask = cv2.add(all_objects_mask, mask) # Aggiunge la maschera di ogni componente


        # Crea un'immagine che è una copia dell'originale
        img_without_objects = img_original_gray.copy()

        # Imposta a nero i pixel nell'immagine copiata dove c'è un oggetto nella maschera
        # Usiamo la maschera all_objects_mask come indice
        img_without_objects[all_objects_mask > 0] = 0 # Mette a 0 i pixel dove la maschera è > 0 (bianca)

        # --- Normalizza l'immagine senza gli oggetti ---
        # Converti il tipo di dato in float32
        img_without_objects_normalized = img_without_objects.astype(np.float32)

        # Scala i valori dei pixel nell'intervallo [0, 1]
        # Dividi per il valore massimo possibile per il tipo di dato originale (255 per uint8)
        img_without_objects_normalized = img_without_objects_normalized / 255.0

        percentage = 0.5  # Ad esempio, il 99%
        max_value = np.max(img_without_objects_normalized)  # Trova il valore massimo nell'immagine
        threshold_value = percentage * max_value
        img_without_objects_normalized[img_without_objects_normalized < threshold_value] = 0
        # --- Crea il negativo dell'immagine normalizzata ---
        img_negative_normalized = 1.0 - img_without_objects_normalized

        # --- Visualizza l'immagine normalizzata e il suo negativo ---
        if plot:
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)  # 1 riga, 2 colonne, primo plot
            plt.imshow(img_without_objects_normalized, cmap='gray')
            plt.title("Immagine Originale senza Oggetti (Normalizzata)")
            plt.axis('off')

            plt.subplot(1, 2, 2)  # 1 riga, 2 colonne, secondo plot
            plt.imshow(img_negative_normalized, cmap='gray')
            plt.title("Negativo (Normalizzata)")
            plt.axis('off')

            plt.show()

    else:
        print("Errore durante la prima decomposizione dell'immagine originale.")

    return img_without_objects_normalized