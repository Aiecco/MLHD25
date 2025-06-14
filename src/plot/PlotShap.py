import cv2
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf

def explain_prediction_shap(model, image_tensor, original_image_for_display, save_path=None):
    """
    Explains a single prediction using SHAP (SHapley Additive exPlanations).
    Requires 'shap' library to be installed (pip install shap).
    This is a conceptual structure; SHAP for images (DeepExplainer) can be memory intensive.

    Args:
        model (tf.keras.Model): The trained Keras model.
        image_tensor (tf.Tensor): The preprocessed input image tensor (1, H, W, C) for the model.
        original_image_for_display (np.array): The original (denormalized, 0-1) image for display.
        save_path (str, optional): Path to save the plot.
    """
    try:
        import shap
    except ImportError:
        print("SHAP library not installed. Please run 'pip install shap'.")
        return

    # SHAP DeepExplainer needs a background dataset for reference.
    # This can be a small subset of your training data.
    # For simplicity, we use a single blank image or averaged image as background.
    # In a real scenario, use a sample of your training data (e.g., 10-50 images).
    background_image = tf.zeros_like(image_tensor)  # Or use a mean image from your dataset
    # If using a dataset, it might look like:
    # background_data = next(iter(train_dataset.take(num_background_samples)))[0]
    # background = background_data.numpy()

    # The DeepExplainer takes (model, data). 'data' should be the background.
    # For Keras models, it automatically detects the input layer.
    explainer = shap.DeepExplainer(model, background_image.numpy())

    # Calculate SHAP values for the input image
    # Note: This can be computationally intensive for large images/models
    shap_values = explainer.shap_values(image_tensor.numpy())

    # SHAP values for images are usually per-channel. For grayscale, it's simpler.
    # `shap_values` for regression will be a list with one array, matching output shape.
    # `shap_values[0]` will have shape (1, H, W, C)

    # Reshape for plotting. Assumes single output (regression).
    if isinstance(shap_values, list) and len(shap_values) == 1:
        shap_values = shap_values[0]  # Take the array for the single output

    # Plot the SHAP values
    # `shap.image_plot` expects image to be 0-1 or 0-255.
    # original_image_for_display should be 0-1
    shap_image_input = (original_image_for_display * 255).astype(np.uint8)
    if len(shap_image_input.shape) == 2:  # SHAP expects 3 channels usually
        shap_image_input = cv2.cvtColor(shap_image_input, cv2.COLOR_GRAY2RGB)
    elif shap_image_input.shape[-1] == 1:
        shap_image_input = cv2.cvtColor(shap_image_input, cv2.COLOR_GRAY2RGB)

    plt.figure(figsize=(8, 4))
    shap.image_plot(shap_values, shap_image_input, show=False)  # show=False to handle figure saving
    plt.suptitle('SHAP Explanation for Predicted Age', y=1.02, fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()