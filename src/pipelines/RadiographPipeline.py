import os.path

import keras

from src.dataset.RadiographDataset import RadiographDatasetBuilder
import tensorflow as tf

# from src.plot.PlotImages import display_raw_prep
from src.preprocessing.PreprocessPipeline import preprocess_pipeline
from src.testing.EvaluationPipeline import evaluation_pipeline
from src.testing.evaluate import evaluate_saved_model
from src.training.RadiographTraining import train_model
from src.training.training_pipeline import training_pipeline


# from src.training.RadiographTraining import train_model

MODEL_PATH = 'best_age_prediction_model_standalone.keras'


def radiograph_pipeline(preprocess=False, training=False, evaluate=False):
    # Estrazione radiomiche
    if preprocess:
        preprocess_pipeline('data/Val/', Val=True)
        preprocess_pipeline('data/Test/', Test=True, Val=False)
        preprocess_pipeline('data/Train/', Train=True, Val=False)

    if training:
        training_pipeline(base_dir_train='data/Train',
                          label_train='data/Train/train_labels.csv',
                          label_val='data/Val/val_labels.csv',
                          base_dir_val='data/Val'
                          )

    if evaluate:
        evaluation_pipeline(MODEL_PATH, test_path='data/Test', label_path='data/Test/test_labels.csv')