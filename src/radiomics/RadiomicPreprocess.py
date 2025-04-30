import os

from src.radiomics.RadiomicsFeature import extract_radiomics_tf


# Elabora e salva le radiomics per tutte le immagini nella directory specificata
def preprocess_radiomics(base_dir, output_dir):
    """
    Elabora le immagini nella directory base, estrae le radiomiche e le salva.
    """
    image_names = [img for img in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, img))]

    for img_name in image_names:
        image_path = os.path.join(base_dir, img_name)
        filename_base = os.path.splitext(img_name)[0]  # Ottieni il nome del file senza l'estensione
        extract_radiomics_tf(image_path, filename_base, output_dir)
