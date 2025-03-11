import os

from src.Models.DualBranchCNN import DualBranchCNN
from src.Utils.save_tensors import save_tensors
from src.preprocessing.preprocess import preprocess_dataset
from src.preprocessing.preprocess_images import plot_tensor_image


def pipeline_dualbranchCNN(preprocess=False):

    # Per le cartelle in data/
    test_path = 'data/Test/test_samples'
    train_path = 'data/Train/train_samples'
    val_path = 'data/Val/validation_samples'

    test_tensors = []
    train_tensors = []
    val_tensors = []

    if preprocess:
        # Processa la cartella di test
        print('Preprocessing dataset:\nTest set')
        for filename in os.listdir(test_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # Filtra i file immagine
                image_path = os.path.join(test_path, filename)
                test_tensors.append(preprocess_dataset(image_path))
        if test_tensors is not None:
            # Esempio: plotta il primo tensore (se presente)
            plot_tensor_image(test_tensors[0])
            save_tensors(os.path.join(test_path, '../tensors'), test_tensors)

        # Processa la cartella di training
        print('Training set')
        for filename in os.listdir(train_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(train_path, filename)
                train_tensors.append(preprocess_dataset(image_path))
        if train_tensors is not None:
            plot_tensor_image(train_tensors[0])
            save_tensors(os.path.join(test_path, '../tensors'), train_tensors)

        print('Validation set')
        # Processa la cartella di validazione
        for filename in os.listdir(val_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(train_path, filename)
                val_tensors.append(preprocess_dataset(image_path))

        if val_tensors is not None:
            plot_tensor_image(val_tensors[0])
            save_tensors(os.path.join(val_path, '../tensors'), val_tensors)


