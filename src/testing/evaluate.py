import os

import numpy as np
import tensorflow as tf
from keras import models
from typing import Tuple, List, Optional

from src.Models.AttentionLayer import SpatialAttention
from src.dataset.RadiographDataset import RadiographDatasetBuilder
from src.utils.interpretPipeline import run_all_interpretability_plots

def evaluate_saved_model(model_path: str,
                         test_base_dir: str,
                         label_test_dataset_path: str,
                         mean_pixel_value: float,
                         std_pixel_value: float,
                         img_sizes: int = 256) -> Tuple[
    Optional[List[float]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Loads a saved Keras model, evaluates it on a test dataset, collects
    the true labels and corresponding predictions, and performs interpretability visualizations.

    Args:
        model_path (str): The file path to the saved Keras model
                          (e.g., '.keras', '.h5', or SavedModel directory).
        test_base_dir (str): The base directory for the test dataset,
                             expected to contain a 'prep_images' subfolder
                             and the label CSV.
        label_test_dataset_path (str): The filename of the CSV file containing
                                       ground truth labels for the test set,
                                       relative to `test_base_dir`.
        mean_pixel_value (float): The mean pixel value calculated from the TRAINING dataset,
                                  used for standardizing image data. Crucial for consistency.
        std_pixel_value (float): The standard deviation pixel value calculated from the TRAINING dataset,
                                 used for standardizing image data. Crucial for consistency.
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

    # Initialize the RadiographDatasetBuilder for the test set.
    # It ensures images are loaded and standardized consistently using the provided mean/std
    # values (which should come from the training set).
    builder_test = RadiographDatasetBuilder(
        base_dir=test_base_dir,
        label_csv=label_test_dataset_path,
        mean_pixel_value=mean_pixel_value,  # Pass the mean value for standardization
        std_pixel_value=std_pixel_value,    # Pass the standard deviation for standardization
        img_size=(img_sizes, img_sizes),
        batch_size=1  # Set batch size to 1 to easily capture a single sample for visualizations
    )
    # Build the TensorFlow Dataset for testing. `shuffle=False` ensures consistent order.
    test_dataset = builder_test.build(train=False)

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

    # Variables to capture a single sample for interpretability visualizations
    sample_preprocessed_image_tensor = None
    sample_true_age = None

    try:
        # Iterate over the test dataset to collect true labels and generate predictions.
        # Batch size is 1, so each `inputs` and `labels` tensor represents a single sample.
        for i, (inputs, labels) in enumerate(test_dataset):
            # Capture the first sample for visualizations.
            # `inputs` will already have a batch dimension of 1 due to `batch_size=1`.
            if sample_preprocessed_image_tensor is None:
                sample_preprocessed_image_tensor = inputs # This is already (1, H, W, C)
                sample_true_age = labels.numpy()[0] # Extract scalar true age

            # Generate prediction for the current single input.
            # `verbose=0` suppresses output during prediction.
            # `[0][0]` is used to extract the scalar prediction value from the output tensor.
            prediction = loaded_model.predict(inputs, verbose=0)[0][0]

            # Append the numpy value of the true label and the scalar prediction.
            true_months.append(labels.numpy()[0]) # Extract scalar true age
            pred_months.append(prediction)

            # Optional: If you only need interpretability for the very first sample,
            # you could break here to speed up evaluation for large datasets.
            # However, typically you want to evaluate the whole test set.
            # if i == 0:
            #     # If you want to stop after processing just the first sample for visualization,
            #     # you might need to adjust how the overall evaluation `loaded_model.evaluate` is called
            #     # or ensure this loop processes the whole dataset.
            #     pass
            # else:
            #     # If you uncomment this, only the first image will be processed for both eval and viz.
            #     # For full dataset evaluation, remove this `break`.
            #     # break
            pass # Keep processing the full dataset for comprehensive evaluation

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
    # if it has already been consumed by the previous loop, to get a full evaluation.
    test_dataset_for_eval = builder_test.build(train=False)  # Rebuild to ensure full evaluation
    results = loaded_model.evaluate(test_dataset_for_eval, verbose=0)

    print("Evaluation Results:")
    # Print each metric name and its corresponding value from the evaluation results.
    for name, value in zip(loaded_model.metrics_names, results):
        print(f"{name}: {value:.4f}")

    # --- Execute interpretability visualizations on the captured sample ---
    if sample_preprocessed_image_tensor is not None:
        run_all_interpretability_plots(
            loaded_model,
            sample_preprocessed_image_tensor,
            sample_true_age,
            mean_pixel_value,  # Pass the mean value for denormalization
            std_pixel_value    # Pass the std value for denormalization
        )
    else:
        print("Could not retrieve a sample image for interpretability visualizations. Skipping.")

    return results, true_months_np, pred_months_np