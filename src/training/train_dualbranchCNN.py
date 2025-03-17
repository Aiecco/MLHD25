import os
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
import matplotlib.pyplot as plt
import numpy as np

from src.Utils.filelabels_search import filelabels_search
from src.preprocessing.preprocess_images import preprocess_pooled_and_heatmap
from src.Models.DualBranchCNN import DualBranchCNN


def process_folder(folder_path, tensors_dict):
    print(f'Preprocessing dataset:\n{os.path.basename(folder_path)} set')
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            id_img = int(filename.split('.')[0])
            image_path = os.path.join(folder_path, filename)
            tensors_dict[id_img] = {}  # creo un dizionario per ogni id_img
            tensors_dict[id_img]['tensor'], tensors_dict[id_img]['heated'] = preprocess_dataset(image_path)
            tensors_dict[id_img]['tensor'] = tf.convert_to_tensor(tensors_dict[id_img]['tensor'], dtype=tf.float32)
            tensors_dict[id_img]['heated'] = tf.convert_to_tensor(tensors_dict[id_img]['heated'], dtype=tf.float32)

    return tensors_dict


def train_model(model, train_files_path='data/Train/tensors', val_files_path='data/Val/tensors', num_epochs=20,
                batch_size=32, learning_rate=1e-3, save_path="dual_branch_cnn_model.h5"):
    """
    Funzione per addestrare il modello e salvare il modello addestrato (TensorFlow) con validation.
    """

    # Carica i dati di training
    train_files = [f for f in os.listdir(train_files_path) if f.endswith('.pt')]
    train_data = {"pooled": [], "heatmap": [], "gender": [], "age": []}
    labels_path = os.path.join(train_files_path, "..", "train_labels.csv")
    heatmaps_path = os.path.join(train_files_path, "..", "heatmaps")
    for file in train_files:
        file_path = os.path.join(train_files_path, file)
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
        train_data["pooled"].append(pooled_tensor)
        train_data["heatmap"].append(heatmap_tensor)
        train_data["gender"].append(gender)
        train_data["age"].append(age)
    gender_mapping = {"M": 1, "F": 0}
    train_data["gender"] = [gender_mapping[g] if isinstance(g, str) else g for g in train_data["gender"]]
    train_pooled_tensors = tf.stack(train_data["pooled"])
    train_heatmap_tensors = tf.stack(train_data["heatmap"])
    train_gender_tensors = tf.convert_to_tensor(train_data["gender"], dtype=tf.float32)
    train_age_tensors = tf.convert_to_tensor(train_data["age"], dtype=tf.float32)
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_pooled_tensors, train_heatmap_tensors, train_gender_tensors, train_age_tensors))
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).batch(batch_size)

    # Carica i dati di validation
    val_files = [f for f in os.listdir(val_files_path) if f.endswith('.pt')]
    val_data = {"pooled": [], "heatmap": [], "gender": [], "age": []}
    val_labels_path = os.path.join(val_files_path, "..", "train_labels.csv")
    val_heatmaps_path = os.path.join(val_files_path, "..", "heatmaps")
    for file in val_files:
        file_path = os.path.join(val_files_path, file)
        try:
            id_img = int(os.path.splitext(file)[0])
        except ValueError:
            print(f"Nome file non valido, saltato: {file}")
            continue
        pooled_tensor = tf.convert_to_tensor(np.load(file_path), dtype=tf.float32)
        heatmap_file = f"{id_img}.pt"
        heatmap_tensor_path = os.path.join(val_heatmaps_path, heatmap_file)
        if not os.path.exists(heatmap_tensor_path):
            print(f"Heatmap non trovata per l'immagine {id_img}, saltata.")
            continue
        heatmap_tensor = tf.convert_to_tensor(np.load(heatmap_tensor_path), dtype=tf.float32)
        gender, age = filelabels_search(val_labels_path, id_img)
        val_data["pooled"].append(pooled_tensor)
        val_data["heatmap"].append(heatmap_tensor)
        val_data["gender"].append(gender)
        val_data["age"].append(age)
    val_data["gender"] = [gender_mapping[g] if isinstance(g, str) else g for g in val_data["gender"]]
    val_pooled_tensors = tf.stack(val_data["pooled"])
    val_heatmap_tensors = tf.stack(val_data["heatmap"])
    val_gender_tensors = tf.convert_to_tensor(val_data["gender"], dtype=tf.float32)
    val_age_tensors = tf.convert_to_tensor(val_data["age"], dtype=tf.float32)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (val_pooled_tensors, val_heatmap_tensors, val_gender_tensors, val_age_tensors))
    val_dataset = val_dataset.batch(batch_size)

    # Definizione dell'ottimizzatore e della funzione di loss
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    loss_fn = losses.MeanSquaredError()

    # Liste per salvare loss e errore percentuale medio di ogni epoch
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_train_percent_errors = []
    epoch_val_percent_errors = []

    # Abilita il plotting interattivo
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Errore Percentuale Medio (%)")
    ax.set_title("Andamento dell'Errore Percentuale Medio durante l'addestramento")

    print("Inizio training...")
    for epoch in range(1, num_epochs + 1):
        # Training
        running_train_loss = 0.0
        running_train_percent_error = 0.0
        for pooled_input, heatmap_input, gender_input, age_target in train_dataset:
            with tf.GradientTape() as tape:
                outputs = model(pooled_input, heatmap_input, gender_input, training=True)
                loss = loss_fn(age_target, tf.squeeze(outputs))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            running_train_loss += loss.numpy() * pooled_input.shapeoled_input.shape[0]

    # Salva il modello alla fine del training
    model.save(save_path)
    print(f"Modello salvato in {save_path}")

    return epoch_train_losses, epoch_val_losses, epoch_train_percent_errors, epoch_val_percent_errors