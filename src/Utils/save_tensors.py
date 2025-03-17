import os
import tensorflow as tf

def save_tensors(image_path, heated_path, id_img, data):
    if data is None:
        return
    try:
        # Crea la cartella principale se non esiste
        os.makedirs(image_path, exist_ok=True)
        os.makedirs(heated_path, exist_ok=True)

        # Crea le variabili TensorFlow
        tensor_variable = tf.Variable(data["tensor"], name=f"tensor_{id_img}")
        heated_variable = tf.Variable(data["heated"], name=f"heated_{id_img}")

        # Salva le variabili utilizzando tf.train.Checkpoint
        checkpoint = tf.train.Checkpoint(tensor=tensor_variable, heated=heated_variable)
        checkpoint_path_tensor = os.path.join(image_path, f"tensor_{id_img}")
        checkpoint_path_heated = os.path.join(heated_path, f"heated_{id_img}")

        checkpoint.write(checkpoint_path_tensor)
        checkpoint.write(checkpoint_path_heated)

        print(f"Tensore salvato come variabile TensorFlow in: {checkpoint_path_tensor} e {checkpoint_path_heated}")

    except Exception as e:
        print(f"Errore durante il salvataggio dei dati: {e}")