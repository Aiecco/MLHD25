import os
import torch
from src.Models.DualBranchCNN import DualBranchCNN
from src.preprocessing.preprocess import preprocess

def pipeline_dualbranchCNN():

    # Per le cartelle in data/
    test_path = '../../data/Test/'
    train_path = '../../data/Train/'
    val_path = '../../data/Val/'

    for test_image, train_image, val_image in zip(test_path, train_path, val_path):
        test_images = preprocess(test_image)
        train_images = preprocess(train_image)
        val_images = preprocess(val_image)



    # Parametri di esempio
    batch_size = 8
    img_size = (128, 128)
    input_channels = 1
    gender_dim = 2  # per esempio, se usi one-hot per 2 generi

    # Inizializzazione del modello
    model = DualBranchCNN(input_channels=input_channels, img_size=img_size, gender_dim=gender_dim)

    # Forward pass
    age_pred = model(pooled_images, heatmaps, gender_data)
    print("Predicted ages:", age_pred.shape)