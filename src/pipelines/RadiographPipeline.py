import os.path

# Imports for preprocessing utilities
from src.preprocessing.PreprocessImage import calculate_mean_std  # Function to calculate mean/std for standardization
from src.preprocessing.PreprocessPipeline import preprocess_pipeline  # Full preprocessing pipeline function

# Imports for model evaluation
from src.testing.EvaluationPipeline import evaluation_pipeline  # Orchestrates model evaluation

# Imports for model training
from src.training.training_pipeline import training_pipeline  # Orchestrates model training pipeline

# Imports for utility functions
from src.utils.load_model import load_trained_model  # Utility function to load a Keras model

# --- Global Constants ---
MODEL_PATH = 'best_age_prediction_model.keras'
"""
str: The default file path for saving and loading the best trained Keras model.
"""
TARGET_IMG_SIZE = 256
"""
int: The target size (height and width) in pixels for preprocessed images.
     Images will be resized to (TARGET_IMG_SIZE, TARGET_IMG_SIZE).
"""


def radiograph_pipeline(preprocess: bool = False, training: bool = False, evaluate: bool = False):
    """
    Main pipeline orchestrator for bone age assessment from radiographs.

    This function controls the entire workflow, including data preprocessing,
    model training, and model evaluation, based on the boolean flags provided.

    Args:
        preprocess (bool, optional): If True, executes the image preprocessing pipeline
                                     for training, validation, and test datasets.
                                     This creates standardized image files. Defaults to False.
        training (bool, optional): If True, executes the model training pipeline.
                                   It calculates mean/std for standardization, loads an
                                   existing model (if available), and initiates training.
                                   Defaults to False.
        evaluate (bool, optional): If True, executes the model evaluation pipeline on
                                   the test dataset using the best saved model. Defaults to False.
    """
    # --- Phase 1: Data Preprocessing ---
    # This block performs image preprocessing (hand detection, CLAHE, resizing, augmentation)
    # and saves the processed images to specified 'prep_images' directories.
    # It needs to be run once to prepare the dataset for training/evaluation.
    if preprocess:
        print("\n--- Starting Data Preprocessing ---")
        preprocess_pipeline('data/Val/', Val=True)
        preprocess_pipeline('data/Test/', Test=True, Val=False)  # Val=False ensures it's treated as Test
        preprocess_pipeline('data/Train/', Train=True, Val=False)  # Val=False ensures it's treated as Train
        print("--- Data Preprocessing Completed ---")

    # --- Phase 2: Model Training ---
    # This block handles the training of the deep learning model.
    # It calculates standardization parameters, attempts to load a previously
    # trained model to continue training, and then runs the training pipeline.
    if training:
        print("\n--- Starting Model Training ---")
        # Define base directories and label CSVs for training and validation datasets.
        # These paths assume a specific directory structure for your data.
        base_dir_train = 'data/Train'
        base_dir_val = 'data/Val'
        label_train_csv = os.path.join(base_dir_train, 'train_labels.csv')  # Assuming specific CSV names
        label_val_csv = os.path.join(base_dir_val, 'val_labels.csv')

        # Calculate mean and standard deviation of pixel values from the preprocessed
        # training images. These values are crucial for standardizing all datasets
        # (train, validation, test) to ensure consistency during model inference.
        train_prep_images_path = os.path.join(base_dir_train, 'prep_images')
        mean_val, std_val = calculate_mean_std(train_prep_images_path, img_size=(TARGET_IMG_SIZE, TARGET_IMG_SIZE))

        # Attempt to load a pre-trained model. If MODEL_PATH exists and the model
        # can be loaded, training will resume from its current state. Otherwise,
        # a new model will be initialized from scratch.
        model = load_trained_model(model_save_path=MODEL_PATH)

        # Execute the training pipeline.
        # It orchestrates dataset loading, model compilation, and the training loop
        # using the calculated standardization values and the loaded/new model.
        training_pipeline(base_dir_train=base_dir_train,
                          label_train=label_train_csv,
                          label_val=label_val_csv,
                          base_dir_val=base_dir_val,
                          img_sizes=TARGET_IMG_SIZE,
                          mean_pixel_value=mean_val,
                          std_pixel_value=std_val,
                          existing_model=model,  # Pass the loaded model (or None if new)
                          model_save_path=MODEL_PATH)
        print("--- Model Training Completed ---")

    # --- Phase 3: Model Evaluation ---
    # This block evaluates the performance of the best saved model on the test dataset.
    # It uses the same standardization parameters from training for consistency.
    if evaluate:
        print("\n--- Starting Model Evaluation ---")
        # Define base directories and label CSV for the test dataset.
        base_dir_test = 'data/Test'
        label_test_csv = os.path.join(base_dir_test, 'test_labels.csv')

        # Execute the evaluation pipeline.
        # It loads the best saved model and evaluates it on the test dataset,
        # applying the same standardization used during training.
        evaluation_pipeline(model_save_path=MODEL_PATH,
                            test_path=base_dir_test,
                            label_path=label_test_csv,
                            img_sizes=TARGET_IMG_SIZE)
        print("--- Model Evaluation Completed ---")
