import os
import tensorflow as tf
from tensorflow.keras import losses

from src.Utils.filelabels_search import filelabels_search
from src.Utils.load_tensors import load_tensor


def test_model(model, test_files_path='data/Test/tensors', batch_size=32):
    """
    Funzione per testare il modello e calcolare l'errore percentuale medio, la loss e l'errore medio in mesi sui dati di test.
    """

    # Carica i dati di test
    test_files = [f for f in os.listdir(test_files_path) if f.endswith('.data-00000-of-00001')]
    test_data = {"pooled": [], "heatmap": [], "gender": [], "age": []}
    test_labels_path = os.path.join(test_files_path, "..", "test_labels.csv")
    for file in test_files:
        file_prefix = os.path.join(test_files_path, file.split(".data")[0])
        try:
            id_img = int(file.split("_")[1].split(".data")[0])
        except ValueError:
            print(f"Nome file non valido, saltato: {file}")
            continue
        pooled_tensor = load_tensor(file_prefix.replace("heatmaps", "tensors"), "tensor/.ATTRIBUTES/VARIABLE_VALUE")
        heatmap_tensor = load_tensor(file_prefix, "heated/.ATTRIBUTES/VARIABLE_VALUE")

        gender, age = filelabels_search(test_labels_path, id_img)
        test_data["pooled"].append(tf.transpose(pooled_tensor, perm=[1, 2, 0]))
        test_data["heatmap"].append(tf.transpose(heatmap_tensor, perm=[1, 2, 0]))
        test_data["gender"].append(gender)
        test_data["age"].append(age)

    gender_mapping = {"M": 1, "F": 0}
    test_data["gender"] = [gender_mapping[g] if isinstance(g, str) else g for g in test_data["gender"]]
    test_pooled_tensors = tf.stack(test_data["pooled"])
    test_heatmap_tensors = tf.stack(test_data["heatmap"])
    test_gender_tensors = tf.convert_to_tensor(test_data["gender"], dtype=tf.float32)

    test_age_corrected = [float(s.replace(',', '.')) for s in test_data["age"]]
    test_age_tensors = tf.convert_to_tensor(test_age_corrected, dtype=tf.float32)

    test_dataset = tf.data.Dataset.from_tensor_slices(
        (test_pooled_tensors, test_heatmap_tensors, test_gender_tensors, test_age_tensors))
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

    # Calcola la loss, l'errore percentuale medio e l'errore medio in mesi sui dati di test
    loss_fn = losses.MeanSquaredError()
    total_loss = 0.0
    total_percent_error = 0.0
    total_absolute_error = 0.0  # Errore assoluto totale
    num_batches = 0

    for pooled_input, heatmap_input, gender_input, age_target in test_dataset:
        outputs = model(pooled_input, heatmap_input, gender_input, training=False)
        loss = loss_fn(age_target, tf.squeeze(outputs))
        total_loss += loss.numpy() * pooled_input.shape[0]

        age_target_np = age_target.numpy()
        outputs_np = tf.squeeze(outputs).numpy()
        percent_errors = abs((age_target_np - outputs_np) / age_target_np) * 100
        total_percent_error += tf.reduce_mean(percent_errors).numpy() * pooled_input.shape[0]
        total_absolute_error += tf.reduce_sum(abs(age_target_np - outputs_np))
        num_batches += pooled_input.shape[0]

    avg_loss = total_loss / num_batches
    avg_percent_error = total_percent_error / num_batches
    avg_absolute_error = total_absolute_error / num_batches
    avg_months_error = avg_absolute_error * 12  # Converte l'errore medio in mesi

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Errore Percentuale Medio: {avg_percent_error:.2f}%")
    print(f"Test Errore Medio in Mesi: {avg_months_error:.2f} mesi")