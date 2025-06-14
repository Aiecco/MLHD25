import numpy as np
from keras import Model
from matplotlib import pyplot as plt
import tensorflow as tf

def visualize_filters(model, layer_name, filter_indices=None, iterations=50, learning_rate=0.1, save_path=None):
    """
    Generates and visualizes patterns that maximally activate specific filters in a given layer.
    This is a simplified conceptual example; robust implementations often use dedicated libraries.

    Args:
        model (tf.keras.Model): The trained Keras model.
        layer_name (str): The name of the convolutional layer to visualize filters from.
        filter_indices (list, optional): List of specific filter indices to visualize.
                                        If None, visualize a few representative filters.
        iterations (int): Number of gradient ascent steps.
        learning_rate (float): Step size for gradient ascent.
        save_path (str, optional): Path to save the plot.
    """
    try:
        layer = model.get_layer(layer_name)
        if not isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)):
            print(f"Layer '{layer_name}' is not a convolutional layer. Cannot visualize filters.")
            return
    except ValueError:
        print(f"Error: Layer '{layer_name}' not found in the model.")
        return

    # Select filters to visualize
    if filter_indices is None:
        n_filters_total = layer.filters  # Number of filters in the layer
        filter_indices = np.linspace(0, n_filters_total - 1, min(8, n_filters_total), dtype=int)

    # Placeholder for generated images
    generated_images = []

    for filter_index in filter_indices:
        # We want to maximize the activation of a specific filter
        # at a specific spatial location (e.g., center) of the output feature map.
        # This requires a 'feature_extractor' model that outputs the activations of the target layer.
        feature_extractor = Model(inputs=model.input, outputs=layer.output)

        # Initialize a random input image
        # Needs to match your model's input shape (e.g., (1, H, W, C))
        # Start with small random noise
        img_height, img_width, img_channels = model.input_shape[1:]
        input_img_data = tf.random.uniform(shape=(1, img_height, img_width, img_channels), minval=0.0, maxval=1.0)
        input_img_data = tf.Variable(input_img_data)  # Make it a tf.Variable for gradient optimization

        for _ in range(iterations):
            with tf.GradientTape() as tape:
                tape.watch(input_img_data)
                # Get the activations of the chosen layer
                layer_activation = feature_extractor(input_img_data)
                # Maximize the activation of the chosen filter
                # We often take the mean of the filter's output for robustness
                loss = tf.reduce_mean(layer_activation[:, :, :, filter_index])

            # Compute gradients of the loss with respect to the input image
            grads = tape.gradient(loss, input_img_data)
            # Normalize the gradients
            grads = grads / (tf.norm(grads) + 1e-8)  # Add epsilon to avoid div by zero

            # Update the input image using gradient ascent
            input_img_data.assign_add(grads * learning_rate)

            # Optional: apply constraints (e.g., pixel values within 0-1 range)
            input_img_data.assign(tf.clip_by_value(input_img_data, 0.0, 1.0))

        # Post-process the generated image for display
        generated_image = input_img_data.numpy().squeeze()  # Remove batch dim and channel if 1
        generated_images.append(generated_image)

    # Plotting
    plt.figure(figsize=(len(filter_indices) * 2, 3))
    plt.suptitle(f'Visualized Filters for Layer: {layer_name}', y=1.02, fontsize=14)
    for i, img in enumerate(generated_images):
        ax = plt.subplot(1, len(filter_indices), i + 1)
        # Assuming grayscale images for your X-rays
        plt.imshow(img, cmap='gray')
        plt.title(f'Filter {filter_indices[i]}', fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
