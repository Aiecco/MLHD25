# Funzione per salvare i tensori in una cartella
import os
import torch


def save_tensors(image_path, tensors):
    if tensors is None:
      return
    try:
        # Crea la cartella "tensors" se non esiste
        tensors_folder = os.path.join(image_path)
        os.makedirs(tensors_folder, exist_ok=True)

        # Genera un nome di file univoco
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        tensor_file_path = os.path.join(tensors_folder, f"{image_name}.pt")

        # Salva il tensore
        torch.save(tensors, tensor_file_path)
        print(f"Tensore salvato in: {tensor_file_path}")

    except Exception as e:
        print(f"Errore durante il salvataggio del tensore: {e}")