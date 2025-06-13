import os
from keras.models import load_model
import keras
from typing import Optional  # Import for Optional type hint

# Assuming correct import path for your custom AttentionLayer
from src.Models.AttentionLayer import SpatialAttention

def load_trained_model(model_save_path: str) -> Optional[keras.Model]:
    """
    Attempts to load a pre-trained Keras model from a specified path.

    This utility function checks for the existence of the model file and
    handles potential errors during the loading process. It's designed
    to facilitate resuming training from a checkpoint or using a saved model
    for evaluation.

    Args:
        model_save_path (str): The file path to the saved Keras model
                               (e.g., '.keras', '.h5', or SavedModel directory).

    Returns:
        Optional[tf.keras.Model]: The loaded Keras model instance if successful.
                                  Returns None if the model file does not exist
                                  or if an error occurs during loading.
    """
    # Check if the model file exists at the specified path.
    if os.path.exists(model_save_path):
        print(f"\nLoading existing model from '{model_save_path}' to continue training...")
        try:
            # Load the Keras model.
            # `custom_objects` is crucial for loading models that use custom layers,
            # like your SpatialAttention layer.
            loaded_model = load_model(
                model_save_path, custom_objects={'SpatialAttention': SpatialAttention}
            )
            print("Model loaded successfully.")
            return loaded_model
        except Exception as e:
            # Catch any exceptions that occur during the model loading process.
            # This can happen if the file is corrupted, the custom object is not found, etc.
            print(f"Error during model loading: {e}")
            import traceback
            traceback.print_exc()  # Print the full stack trace for debugging purposes

            # If loading fails, treat it as if no model was found, and suggest starting anew.
            print("Model loading failed. Proceeding with training a new model.")
            return None  # Indicate failure to load by returning None
    else:
        # Inform the user if the model file does not exist.
        print(f"\nModel not found at '{model_save_path}'. Starting training from scratch.")
        return None  # Indicate that no existing model was found