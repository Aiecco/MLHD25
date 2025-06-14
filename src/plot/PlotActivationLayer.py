from keras import Model
import tensorflow as tf
from matplotlib import pyplot as plt


def visualize_layer_activations(model, image_tensor, layer_names=None, num_filters_to_show=8, save_path=None):
    """
    Visualizes the activations of specified layers for a given input image.

    Args:
        model (tf.keras.Model): The trained Keras model.
        image_tensor (tf.Tensor): The preprocessed input image tensor (batch_size, H, W, C).
        layer_names (list, optional): A list of layer names whose activations should be visualized.
                                      If None, attempts to find representative conv layers.
        num_filters_to_show (int): Number of filters to display per layer.
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    if layer_names is None:
        # Try to find some representative convolutional layers automatically
        layer_names = []
        for layer in model.layers:
            if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D)) and \
                    'attention' not in layer.name and 'output' not in layer.name:  # Avoid attention map output itself
                layer_names.append(layer.name)
        # Select a few, e.g., first, middle, last few conv layers
        if len(layer_names) > 3:
            selected_layers = [layer_names[0],
                               layer_names[len(layer_names) // 2],
                               layer_names[-1]]
            layer_names = selected_layers
        elif len(layer_names) > 0:
            layer_names = layer_names  # Use all if few
        else:
            print("No suitable convolutional layers found for activation visualization.")
            return

    # Create a model that outputs the activations of the specified layers
    outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = Model(inputs=model.input, outputs=outputs)

    # Get activations
    activations = activation_model.predict(image_tensor)

    if not isinstance(activations, list):  # If only one layer was specified
        activations = [activations]

    plt.figure(figsize=(num_filters_to_show * 1.5, len(layer_names) * 1.5 + 2))  # Adjust figure size

    for i, layer_activation in enumerate(activations):
        layer_name = layer_names[i]
        # activations are (1, H, W, Channels) -> take first image and convert to (H, W, Channels)
        activation_map = layer_activation[0]

        # Determine how many filters to show (max available or num_filters_to_show)
        n_filters = min(activation_map.shape[-1], num_filters_to_show)

        for f in range(n_filters):
            ax = plt.subplot(len(layer_names), num_filters_to_show, i * num_filters_to_show + f + 1)
            plt.imshow(activation_map[:, :, f], cmap='viridis')
            plt.axis('off')
            if f == 0:  # Only label the first filter's column
                ax.set_title(f'Layer: {layer_name}\nFilter {f + 1}', fontsize=8)
            else:
                ax.set_title(f'Filter {f + 1}', fontsize=8)

    plt.suptitle('Layer Activations for Input Image', y=0.98, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()