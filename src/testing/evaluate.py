import os

import numpy as np
import tensorflow as tf
from keras import models  # Keras models module for loading saved models
from typing import Tuple, List, Optional  # Imports for type hinting

from src.Models.AttentionLayer import SpatialAttention  # Custom attention layer definition
from src.dataset.RadiographDataset import RadiographDatasetBuilder  # Dataset builder utility
from src.preprocessing.PreprocessImage import calculate_mean_std  # Utility to calculate mean/std for standardization

def evaluate_saved_model(model_path: str,
                         test_base_dir: str,
                         label_test_dataset_path: str,
                         img_sizes: int = 256) -> Tuple[
    Optional[List[float]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads a saved Keras model, evaluates it on a test dataset, and collects
    the true labels and corresponding predictions.

    This function is responsible for ensuring the test data is preprocessed
    consistently with the training data (e.g., using the same mean and
    standard deviation for standardization).

    Args:
        model_path (str): The file path to the saved Keras model
                          (e.g., '.keras', '.h5', or SavedModel directory).
        test_base_dir (str): The base directory for the test dataset,
                             expected to contain a 'prep_images' subfolder
                             and the label CSV.
        label_test_dataset_path (str): The filename of the CSV file containing
                                       ground truth labels for the test set,
                                       relative to `test_base_dir`.
        img_sizes (int, optional): The target square dimension (height and width)
                                   for input images. This must match the size
                                   used during model training and preprocessing.
                                   Defaults to 256.

    Returns:
        Tuple[Optional[List[float]], Optional[np.ndarray], Optional[np.ndarray]]: A tuple containing:
            - results (Optional[List[float]]): A list of evaluation results (e.g., loss, metrics).
                                               Returns None if the model cannot be loaded or evaluated.
            - true_months (Optional[np.ndarray]): A NumPy array of true bone ages in months.
                                                  Returns None if the model cannot be loaded or evaluated.
            - pred_months (Optional[np.ndarray]): A NumPy array of predicted bone ages in months.
                                                  Returns None if the model cannot be loaded or evaluated.
    """
    # Calculate mean and standard deviation from the preprocessed test images.
    # NOTE: It's crucial for consistency that these values (`mean_val`, `std_val`)
    # are derived from the TRAINING set and passed into this function, rather
    # than being recalculated on the test set. Recalculating on the test set
    # can lead to data leakage or inconsistency if the test set's distribution
    # is different from the training set's.
    # For a proper setup, `mean_pixel_value` and `std_pixel_value` should be
    # parameters of this function, derived from the training set.
    # The current implementation calculates them on 'data/Test/prep_images',
    # which assumes 'data/Test' is where the test set's prepared images reside.
    # It also means if this function is called alone, it uses the test set's
    # statistics, which may not be consistent with training.
    mean_val, std_val = calculate_mean_std(os.path.join(test_base_dir, 'prep_images'), img_size=(img_sizes, img_sizes))

    # Initialize the RadiographDatasetBuilder for the test set.
    # It ensures images are loaded and standardized consistently using the calculated mean/std.
    builder_test = RadiographDatasetBuilder(
        base_dir=test_base_dir,
        label_csv=label_test_dataset_path,
        mean_pixel_value=mean_val,  # Pass the calculated mean value for standardization
        std_pixel_value=std_val,  # Pass the calculated standard deviation for standardization
        img_size=(img_sizes, img_sizes),
        batch_size=1  # Set batch size to 1 for individual predictions during evaluation
    )
    # Build the TensorFlow Dataset for testing. `shuffle=False` ensures consistent order.
    test_dataset = builder_test.build(
        shuffle=False)  # Do not shuffle the test set.

    print(f"\nLoading model from: {model_path}")
    try:
        # Load the Keras model from the specified path.
        # `custom_objects` is essential for correctly deserializing custom layers
        # like `SpatialAttention` that are part of your model architecture.
        loaded_model = models.load_model(model_path, custom_objects={'SpatialAttention': SpatialAttention})
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading the model: {e}")
        import traceback  # Import traceback for detailed error logging
        traceback.print_exc()  # Print full stack trace for debugging
        return None, None, None  # Return None values to indicate failure

    print("\nEvaluating and collecting predictions on the Test Set:")

    true_months: List[float] = []  # List to store true bone ages
    pred_months: List[float] = []  # List to store predicted bone ages

    # Iterate over the test dataset to collect true labels and generate predictions.
    # `.unbatch()` is used to process each example individually, which is necessary
    # for collecting one-to-one predictions.
    try:
        for inputs, labels in test_dataset.unbatch():
            # Add a batch dimension to the input tensor as `model.predict` expects batches.
            input_batch = tf.expand_dims(inputs, axis=0)

            # Generate prediction for the single input.
            # `verbose=0` suppresses output during prediction.
            # `[0][0]` is used to extract the scalar prediction value from the output tensor.
            prediction = loaded_model.predict(input_batch, verbose=0)[0][0]

            # Append the numpy value of the true label and the scalar prediction.
            true_months.append(labels.numpy())
            pred_months.append(prediction)
    except tf.errors.OutOfRangeError:
        print("Dataset exhausted (end of iteration).")
    except Exception as e:
        print(f"Error during prediction collection on the test set: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None  # Indicate failure

    # Convert lists of true and predicted values to NumPy arrays for further analysis and plotting.
    true_months_np = np.array(true_months)
    pred_months_np = np.array(pred_months)

    # Perform a formal evaluation of the model on the entire test dataset.
    # It's important to build the test_dataset again or ensure its iterator is reset
    # if it has already been consumed by the `unbatch()` loop, to get a full evaluation.
    test_dataset_for_eval = builder_test.build(shuffle=False)  # Rebuild to ensure full evaluation
    results = loaded_model.evaluate(test_dataset_for_eval, verbose=0)

    print("Evaluation Results:")
    # Print each metric name and its corresponding value from the evaluation results.
    for name, value in zip(loaded_model.metrics_names, results):
        print(f"{name}: {value:.4f}")

    return results, true_months_np, pred_months_np