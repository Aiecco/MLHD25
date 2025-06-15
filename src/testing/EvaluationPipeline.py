import os
import numpy as np
from typing import Tuple  # Import for type hinting

# Assuming these imports are correctly resolved in your project structure
from src.Models.AttentionLayer import SpatialAttention  # Ensure correct casing/path if different
from src.plot.PlotEval import plot_eval  # Assuming plot_eval is a function for plotting evaluation results
from keras import models  # Keras models module for loading saved models

from src.testing.evaluate import evaluate_saved_model


def evaluation_pipeline(model_save_path: str,
                        test_path: str,
                        label_path: str,
                        img_sizes: int,  # Add img_sizes to signature as it's passed to evaluate_saved_model
                        mean_val: float,
                        std_val: float,
                        ):
    """
    Orchestrates the evaluation of a trained deep learning model on a test set
    and generates visual plots of its performance.

    This function loads a previously saved Keras model, evaluates it on the
    specified test dataset (ensuring consistent data preprocessing), and then
    visualizes the prediction errors and distributions.

    Args:
        model_save_path (str): The file path to the saved Keras model (e.g., '.keras', '.h5').
        test_path (str): The base directory of the test dataset containing images.
        label_path (str): The file path to the CSV containing ground truth labels for the test set.
        img_sizes (int): The target square dimension (e.g., 256) of the images used by the model.
                         This must match the size used during model training and preprocessing.
        mean_val (float): The mean pixel value calculated from the TRAINING dataset,
        std_val (float): The standard deviation pixel value calculated from the TRAINING dataset,

    Returns:
        None: This function does not return any value but prints evaluation results
              and saves plots to disk.
    """
    print("\n--- Model Evaluation and Graph Generation ---")

    # Check if the saved model file exists before proceeding.
    if os.path.exists(model_save_path):
        # Evaluate the saved model on the test dataset.
        # It's crucial to pass the mean/std values to ensure the test set is
        # preprocessed identically to the training set.
        evaluation_results, true_months_list, pred_months_list = evaluate_saved_model(
            model_save_path,
            test_path,
            label_path,
            mean_pixel_value=mean_val,
            std_pixel_value=std_val,
            img_sizes=img_sizes,  # Pass img_sizes
        )

        # Load the model again to access its metrics_names.
        # This is redundant with the loading inside evaluate_saved_model
        # but needed here to dynamically get the MAE index if it's not always 1.
        try:
            loaded_model_for_metrics = models.load_model(model_save_path,
                                                         custom_objects={'SpatialAttention': SpatialAttention})
        except Exception as e:
            print(f"Error loading model for metrics names: {e}")
            loaded_model_for_metrics = None

        # Proceed with plotting if evaluation results were successfully obtained.
        if evaluation_results is not None and loaded_model_for_metrics is not None:
            # Dynamically find the index of 'mae' in the model's metrics names.
            # Fallback to index 1 if 'mae' is not explicitly found (assuming loss is at 0, mae at 1).
            mae_index = loaded_model_for_metrics.metrics_names.index(
                'mae') if 'mae' in loaded_model_for_metrics.metrics_names else 1
            mae_months = evaluation_results[mae_index] if mae_index is not None and mae_index < len(
                evaluation_results) else np.nan

            # Calculate absolute errors for plotting the error distribution.
            errors_list = np.abs(true_months_list - pred_months_list)

            # Call the external plotting function to visualize the evaluation results.
            plot_eval(errors_list, mae_months, true_months_list, pred_months_list)
            print("Evaluation graphs generated and saved as 'age_prediction_analysis.png'.")
        else:
            print("\nUnable to evaluate the saved model due to an error during loading or evaluation process.")
    else:
        # Inform the user if the model file was not found.
        print(f"Error: The model was not found at the specified path: {model_save_path}")