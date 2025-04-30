import os
import tensorflow as tf


def save_tensors(image_path, id_img, data):
    if data is None:
        return
    try:
        # Crea la cartella principale se non esiste
        os.makedirs(image_path, exist_ok=True)

        # Crea le variabili TensorFlow
        tensor_variable = tf.Variable(data["tensor"], name=f"tensor_{id_img}")

        # Salva le variabili utilizzando tf.train.Checkpoint
        checkpoint = tf.train.Checkpoint(tensor=tensor_variable)
        checkpoint_path_tensor = os.path.join(image_path, f"tensor_{id_img}")

        checkpoint.write(checkpoint_path_tensor)

        print(f"Tensore salvato come variabile TensorFlow in: {checkpoint_path_tensor}")

    except Exception as e:
        print(f"Errore durante il salvataggio dei dati: {e}")