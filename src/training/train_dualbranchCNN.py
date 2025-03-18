import os
import tensorflow as tf
from tensorflow.keras import optimizers, losses, metrics
import matplotlib.pyplot as plt
import numpy as np

from src.Utils.filelabels_search import filelabels_search
from src.Utils.load_tensors import load_tensor
from src.preprocessing.preprocess import preprocess_dataset
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
                batch_size=32, learning_rate=1e-3, save_path="out/dual_branch_cnn_model.h5"):
    """
    Funzione per addestrare il modello e salvare il modello addestrato (TensorFlow) con validation.
    """

    # Carica i dati di training
    train_files = [f for f in os.listdir(train_files_path) if f.endswith('.data-00000-of-00001')] #cambiato l'estensione
    train_data = {"pooled": [], "heatmap": [], "gender": [], "age": []}
    labels_path = os.path.join(train_files_path, "..", "train_labels.csv")
    heatmaps_path = os.path.join(train_files_path, "..", "heatmaps")
    for file in train_files:
        file_prefix = os.path.join(train_files_path, file.split(".data")[0]) #creo il prefisso corretto
        try:
            id_img = int(file.split("_")[1].split(".data")[0]) #estraggo l'id_img
        except ValueError:
            print(f"Nome file non valido, saltato: {file}")
            continue
        pooled_tensor = load_tensor(file_prefix.replace("heatmaps", "tensors"), "tensor/.ATTRIBUTES/VARIABLE_VALUE")
        heatmap_tensor = load_tensor(file_prefix, "heated/.ATTRIBUTES/VARIABLE_VALUE")

        gender, age = filelabels_search(labels_path, id_img)
        train_data["pooled"].append(tf.transpose(pooled_tensor, perm=[1, 2, 0]))  # added transpose
        train_data["heatmap"].append(tf.transpose(heatmap_tensor, perm=[1, 2, 0]))  # added transpose
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
    train_dataset = train_dataset.shuffle(buffer_size=len(train_files)).batch(batch_size, drop_remainder=True)

    # Carica i dati di validation
    val_files = [f for f in os.listdir(val_files_path) if f.endswith('.data-00000-of-00001')] #cambiato l'estensione
    val_data = {"pooled": [], "heatmap": [], "gender": [], "age": []}
    val_labels_path = os.path.join(val_files_path, "..", "val_labels.csv")
    val_heatmaps_path = os.path.join(val_files_path, "..", "heatmaps")
    for file in val_files:
        file_prefix = os.path.join(val_files_path, file.split(".data")[0])
        try:
            id_img = int(file.split("_")[1].split(".data")[0])
        except ValueError:
            print(f"Nome file non valido, saltato: {file}")
            continue
        pooled_tensor = load_tensor(file_prefix.replace("heatmaps", "tensors"), "tensor/.ATTRIBUTES/VARIABLE_VALUE")
        heatmap_tensor = load_tensor(file_prefix, "heated/.ATTRIBUTES/VARIABLE_VALUE")

        gender, age = filelabels_search(val_labels_path, id_img)
        val_data["pooled"].append(tf.transpose(pooled_tensor, perm=[1, 2, 0]))  # added transpose
        val_data["heatmap"].append(tf.transpose(heatmap_tensor, perm=[1, 2, 0]))  # added transpose
        val_data["gender"].append(gender)
        val_data["age"].append(age)
    val_data["gender"] = [gender_mapping[g] if isinstance(g, str) else g for g in val_data["gender"]]
    val_pooled_tensors = tf.stack(val_data["pooled"])
    val_heatmap_tensors = tf.stack(val_data["heatmap"])
    val_gender_tensors = tf.convert_to_tensor(val_data["gender"], dtype=tf.float32)
    val_age_tensors = tf.convert_to_tensor(val_data["age"], dtype=tf.float32)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (val_pooled_tensors, val_heatmap_tensors, val_gender_tensors, val_age_tensors))
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

    # Definizione dell'ottimizzatore e della funzione di loss
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    loss_fn = losses.MeanSquaredError()

    epoch_train_losses = []
    epoch_val_losses = []

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")

    print("Inizio training...")
    for epoch in range(1, num_epochs + 1):
        # Training
        running_train_loss = 0.0
        for pooled_input, heatmap_input, gender_input, age_target in train_dataset:
            with tf.GradientTape() as tape:
                outputs = model(pooled_input, heatmap_input, gender_input, training=True)
                loss = loss_fn(age_target, tf.squeeze(outputs))
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            running_train_loss += loss.numpy() * pooled_input.shape[0]
        train_loss = running_train_loss / len(train_dataset)
        epoch_train_losses.append(train_loss)

        # Validation
        running_val_loss = 0.0
        for pooled_input, heatmap_input, gender_input, age_target in val_dataset:
            outputs = model(pooled_input, heatmap_input, gender_input, training=False)
            loss = loss_fn(age_target, tf.squeeze(outputs))
            running_val_loss += loss.numpy() * pooled_input.shape[0]
        val_loss = running_val_loss / len(val_dataset)
        epoch_val_losses.append(val_loss)

        # Aggiornamento plot
        ax.clear()
        ax.plot(epoch_train_losses, label='Training Loss')
        ax.plot(epoch_val_losses, label='Validation Loss')
        ax.legend()
        plt.pause(0.1)

        print(f"Epoch {epoch}/{num_epochs}: Training Loss = {train_loss:.4f}, Validation Loss = {val_loss:.4f}")

    model.save(save_path)
    print(f"Modello salvato in {save_path}")
    plt.ioff()
    plt.show()

    return epoch_train_losses, epoch_val_losses