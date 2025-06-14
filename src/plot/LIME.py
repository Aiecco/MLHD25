import cv2
import numpy as np
from matplotlib import pyplot as plt


def explain_prediction_lime(model, image_tensor, original_image_for_display, std_val, mean_val, top_labels=5, save_path=None):
    """
    Explains a single prediction using LIME (Local Interpretable Model-agnostic Explanations).
    Requires 'lime' library to be installed (pip install lime).
    This is a conceptual structure; actual implementation needs careful handling of
    image preprocessing for LIME's internal perturbation.

    Args:
        model (tf.keras.Model): The trained Keras model.
        image_tensor (tf.Tensor): The preprocessed input image tensor (1, H, W, C) for the model.
        original_image_for_display (np.array): The original (denormalized, 0-1) image for display.
        std_val (float): Standard deviation used for internal image preprocessing by LIME.
        mean_val (float): Mean value used for internal image preprocessing by LIME.
        top_labels (int): Number of labels/segments to highlight (for regression, this might be adjusted).
        save_path (str, optional): Path to save the plot.
    """
    try:
        from lime import lime_image
    except ImportError:
        print("LIME library not installed. Please run 'pip install lime'.")
        return

    explainer = lime_image.LimeImageExplainer()

    # Define a prediction function for LIME.
    # LIME expects a function that takes numpy array of images (0-255 RGB)
    # and returns predictions for each image.
    # Your model expects preprocessed (standardized) grayscale images.
    def predict_fn(images):
        # images will be (N, H, W, 3) from LIME, need to convert to (N, H, W, 1) and preprocess
        # Convert RGB to grayscale and then apply your model's preprocessing
        grayscale_images = np.mean(images, axis=-1, keepdims=True) # Simple grayscale conversion
        preprocessed_images = (grayscale_images - mean_val) / std_val
        return model.predict(preprocessed_images)

    # LIME works best for classification. For regression, you might need to
    # treat it as a "class" representing the predicted value or bin predictions.
    # For a simple regression, LIME will try to explain the single output value.
    # The image should be 0-1 or 0-255 range for LIME explainer.
    # original_image_for_display should be 0-1 here, scale to 0-255 if LIME expects it
    lime_image_input = (original_image_for_display * 255).astype(np.uint8)
    if len(lime_image_input.shape) == 2: # LIME expects 3 channels
        lime_image_input = cv2.cvtColor(lime_image_input, cv2.COLOR_GRAY2RGB)
    elif lime_image_input.shape[-1] == 1:
        lime_image_input = cv2.cvtColor(lime_image_input, cv2.COLOR_GRAY2RGB)

    # For regression, LIME will try to explain the single output neuron.
    # 'top_labels' might not be directly applicable as it is for classification.
    # LIME's 'explanation' object for regression provides 'image_and_mask'.
    try:
        explanation = explainer.explain_instance(
            lime_image_input, # Original image (denormalized)
            predict_fn,
            top_labels=1, # For regression, we care about the main output
            hide_color=0, # Color to use for masked regions
            num_samples=1000 # Number of perturbed samples
        )

        # Get image and mask for the top feature (which is the regression output importance)
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], # The single regression output 'label'
            positive_only=True, negative_only=False, num_features=top_labels, hide_rest=True
        )

        plt.figure(figsize=(8, 4))
        plt.imshow(temp / 255.0) # Scale back to 0-1 for matplotlib
        plt.imshow(mask, alpha=0.5, cmap='jet') # Overlay the mask
        plt.title(f'LIME Explanation for Predicted Age')
        plt.axis('off')

        if save_path:
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print(f"Error during LIME explanation: {e}")
        print("Ensure 'original_image_for_display' is properly denormalized (0-1 or 0-255) and LIME's predict_fn is correct for your model's input/output.")
