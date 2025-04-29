import os
import tensorflow as tf
from keras.src.saving import custom_object_scope

from src.Models.DualBranchCNN import DualBranchCNN
from src.Utils.save_tensors import save_tensors
from src.plot.plot_training_progress import plot_training_progress
from src.preprocessing.preprocess import preprocess_dataset
from src.preprocessing.preprocess_images import plot_tensor_image
from src.testing.test_dualbranchCNN import test_model
from src.training.train_dualbranchCNN import train_model


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


def pipeline_dualbranchCNN(preprocess=False, training=False, epochs=10):
    # Percorsi delle cartelle
    test_path = 'data/Test/test_samples'
    train_path = 'data/Train/train_samples'
    val_path = 'data/Val/validation_samples'

    if preprocess:
        test_tensors = process_folder(test_path, {})
        train_tensors = process_folder(train_path, {})
        val_tensors = process_folder(val_path, {})

        # Plot e save per test
        if test_tensors:
            for id_img, data in test_tensors.items():
                if data is not None:
                    # plot_tensor_image(data['tensor'])
                    save_tensors(os.path.join(os.path.dirname(test_path), 'tensors'),
                                  os.path.join(os.path.dirname(test_path), 'heatmaps'), id_img, data)

        # Plot e save per train
        if train_tensors:
            for id_img, data in train_tensors.items():
                if data is not None:
                    # plot_tensor_image(data['tensor'])
                    save_tensors(os.path.join(os.path.dirname(train_path), 'tensors'),
                                  os.path.join(os.path.dirname(train_path), 'heatmaps'), id_img, data)

        # Plot e save per validation
        if val_tensors:
            for id_img, data in val_tensors.items():
                if data is not None:
                    # plot_tensor_image(data['tensor'])
                    save_tensors(os.path.join(os.path.dirname(val_path), 'tensors'),
                                  os.path.join(os.path.dirname(val_path), 'heatmaps'), id_img, data)

    # Inizializza il modello
    model = DualBranchCNN(input_channels=1, img_size=(128, 128))

    # Avvia il training
    if training:
        training_history = train_model(model, num_epochs=epochs, batch_size=32, learning_rate=1e-3)
        #plot_training_progress(epoch_train_losses, epoch_val_losses)

    # test model
    # Carica il modello addestrato
    model_path = "out/dual_branch_cnn_model.keras"

    try:
        # Costruisci modello con stessa architettura
        wrapped_model = DualBranchCNN()

        # Carica SOLO i pesi (serve che tu li abbia salvati con model.save_weights(...))
        wrapped_model.model.load_weights(model_path)
        print(f"Modello caricato da {model_path}")
    except OSError:
        print(f"Errore: Impossibile caricare il modello da {model_path}. Assicurati che il file esista.")
        return

    test_model(wrapped_model)
