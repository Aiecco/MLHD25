import os
import tensorflow as tf
import numpy as np
from tensorflow.keras import losses

from src.Utils.filelabels_search import filelabels_search


def test_model(model, test_files_path='data/Test/tensors', batch_size=32):
    """
    Funzione per testare il modello sui dati di test.
    """
    # Carica i dati di test
    test_files = [f for f in os.listdir(test_files_path) if f.endswith('.pt')]
    test_data = {"pooled": [], "heatmap": [], "gender": [], "age": []}
    labels_path = os.path.join(test_files_path, "..", "train_labels.csv")
    heatmaps_path = os.path.join(test_files_path, "..", "heatmaps")
    for file in test_files:
        file_path = os.path.join(test_files_path, file)
        try:
            id_img = int(os.path.splitext(file)[0])
        except ValueError:
            print(f"Nome file non valido, saltato: {file}")
            continue
        pooled_tensor = tf.convert_to_tensor(np.load(file_path), dtype=tf.float32)
        heatmap_file = f"{id_img}.pt"
        heatmap_tensor_path = os.path.join(heatmaps_path, heatmap_file)
        if not os.path.exists(heatmap_tensor_path):
            print(f"Heatmap non trovata per l'immagine {id_img}, saltata.")
            continue
        heatmap_tensor = tf.convert_to_tensor(np.load(heatmap_tensor_path), dtype=tf.float32)
        gender, age = filelabels_search(labels_path, id_img)
        test_data["pooled"].append(pooled_tensor)
        test_data["heatmap"].append(heatmap_tensor)
        test_data["gender"].append(gender)
        test_data["age"].append(age)
    gender_mapping = {"M": 1, "F": 0}
    test_data["gender"] = [gender_mapping[g] if isinstance(g, str) else g for g in test_data["gender"]]
    test_pooled_tensors = tf.stack(test_data["pooled"])
    test_heatmap_tensors = tf.stack(test_data["heatmap"])
    test_gender_tensors = tf.convert_to_tensor(test_data["gender"], dtype=tf.float32)
    test_age_tensors = tf.convert_to_tensor(test_data["age"], dtype=tf.float32)
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_pooled_tensors, test_heatmap_tensors, test_gender_tensors, test_age_tensors))
    test_dataset = test_dataset.batch(batch_size)

    # Valutazione del modello sui dati di test
    loss_fn = losses.MeanSquaredError()
    total_loss = 0.0
    total_percent_error = 0.0
    num_batches = 0

    for pooled_input, heatmap_input, gender_input, age_target in test_dataset:
        outputs = model(pooled_input, heatmap_input, gender_input, training=False)
        loss = loss_fn(age_target, tf.squeeze(outputs))
        total_loss += loss.numpy() * pooled_input.shape[0]
        percent_errors = tf.abs((tf.squeeze(outputs) - age_target) / age_target) * 100.0
        total_percent_error += tf.reduce_mean(percent_errors).numpy() * pooled_input.shape[0]
        num_batches += pooled_input.shape[0]

    avg_loss = total_loss / len(test_files)
    avg_percent_error = total_percent_error / len(test_files)

    print(f"Test Loss: {avg_loss:.4f}, Errore Percentuale Medio: {avg_percent_error:.2f}%")

