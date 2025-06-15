import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import Model
import cv2  # Per il resizing e la gestione delle immagini
import os

from src.plot.LIME import explain_prediction_lime
from src.plot.PlotActivationLayer import visualize_layer_activations
from src.plot.PlotFilters import visualize_filters
from src.plot.PlotHeatmapOverlay import visualize_attention_map
from src.plot.PlotShap import explain_prediction_shap


# --- Main orchestrator for interpretability plots ---
def run_all_interpretability_plots(
        model,
        preprocessed_image_tensor,
        true_age,
        mean_val,
        std_val,
        output_dir='analysis_plots/interpretability'
):
    """
    Runs all interpretability visualizations for a single image sample.

    Args:
        model (tf.keras.Model): The trained Keras model.
        preprocessed_image_tensor (tf.Tensor): The preprocessed input image tensor (batch_size=1, H, W, C).
        true_age (float): The true age of the patient for the given image.
        mean_val (float): Mean value used for image denormalization.
        std_val (float): Standard deviation used for image denormalization.
        output_dir (str): Directory to save the generated plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Denormalize the original image for LIME/SHAP display purposes (0-1 range)
    # This is the image that will be shown as background for explanation masks.
    original_display_image = (preprocessed_image_tensor[0].numpy() * std_val) + mean_val
    original_display_image = (original_display_image - original_display_image.min()) / \
                             (original_display_image.max() - original_display_image.min() + 1e-8)

    print("\n--- Starting Model Interpretability Visualizations ---")

    # 1. Spatial Attention Map
    print("Generating Spatial Attention Map...")
    visualize_attention_map(
        model,
        preprocessed_image_tensor,
        true_age,
        std_val,
        mean_val,
        save_path=os.path.join(output_dir, 'spatial_attention_map.png')
    )

    # 2. Layer Activations
    print("Generating Layer Activations Visualization...")
    # Adjust layer_names based on your model's architecture
    # You can get a list of layer names by doing `model.summary()`
    visualize_layer_activations(
        model,
        preprocessed_image_tensor,
        layer_names=['prep_conv1a', 'prep_conv3a', 'conv5a'],  # EXAMPLE NAMES - ADJUST FOR YOUR MODEL
        num_filters_to_show=8,
        save_path=os.path.join(output_dir, 'layer_activations.png')
    )

    # 3. Filter Visualization (if you want to include it, ensure layer_name is correct)
    #print("Generating Filter Visualizations...")
    #visualize_filters(
    #     model,
    #     layer_names=['prep_conv1a', 'prep_conv3a', 'conv5a'],
    #     save_path=os.path.join(output_dir, 'filter_visualization.png')
    # )

    # 4. LIME Explanation
    print("Generating LIME Explanation...")
    explain_prediction_lime(
        model,
        preprocessed_image_tensor,
        original_display_image,
        std_val,
        mean_val,
        save_path=os.path.join(output_dir, 'lime_explanation.png')
    )

    # 5. SHAP Explanation
    #print("Generating SHAP Explanation...")
    #explain_prediction_shap(
    #    model,
    #    preprocessed_image_tensor,
    #    original_display_image,
    #    save_path=os.path.join(output_dir, 'shap_explanation.png')
    #)

    print("All interpretability visualizations complete.")