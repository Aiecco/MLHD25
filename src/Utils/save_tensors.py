# Funzione per salvare i tensori in una cartella
import os
import torch


def save_tensors(image_path, id_img, data):
    if data is None:
        return
    try:
        # Crea la cartella principale se non esiste
        os.makedirs(image_path, exist_ok=True)

        # Crea la sottocartella per l'ID
        os.makedirs(image_path, exist_ok=True)

        # Salva il tensore
        tensor_file_path = os.path.join(image_path, f"{id_img}.pt")
        torch.save(data["tensor"], tensor_file_path)
        print(f"Tensore salvato in: {tensor_file_path}")

    except Exception as e:
        print(f"Errore durante il salvataggio dei dati: {e}")
