import tensorflow as tf
from keras import layers, regularizers, models

from src.Models.AttentionLayer import \
    SpatialAttention  # Assuming this import path is correct and SpatialAttention is defined elsewhere

# --- Age Prediction Model with Attention ---
class AgePredictionModel:
    """
    Represents a deep learning model for automated bone age prediction from hand radiographs.

    This model employs a multi-layered Convolutional Neural Network (CNN) backbone
    for feature extraction, integrated with a custom spatial attention mechanism,
    and a robust fully connected regression head. It is designed to be gender-agnostic,
    relying solely on image-derived features for prediction.
    """

    def __init__(self, img_size=(128, 128)):
        """
        Initializes the AgePredictionModel.

        Args:
            img_size (tuple, optional): The target dimensions (height, width) for input images.
                                        Defaults to (128, 128). This should match the size
                                        used in preprocessing.
        """
        self.img_size = img_size
        # Build the Keras model graph immediately upon initialization
        self.model = self._build_model()

    def _create_cnn_branch(self, input_tensor: tf.Tensor, name_prefix: str) -> tf.Tensor:
        """
        Constructs a single Convolutional Neural Network (CNN) branch for feature extraction.

        This branch consists of five sequential blocks, each typically comprising two
        convolutional layers, Batch Normalization, ReLU activation, and MaxPooling.
        The number of filters progressively increases with depth.

        Args:
            input_tensor (tf.Tensor): The input tensor to the CNN branch (e.g., image input).
            name_prefix (str): A prefix for naming the layers within this branch
                               to ensure uniqueness (e.g., 'prep', 'raw', 'extr').

        Returns:
            tf.Tensor: The output tensor from the final MaxPooling layer of this CNN branch,
                       representing extracted spatial features.
        """
        # Block 1: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> MaxPool
        # Captures initial low-level features
        x = layers.Conv2D(32, (3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(1e-4), name=f'{name_prefix}_conv1a')(input_tensor)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1a')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu1a')(x)
        x = layers.Conv2D(32, (3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(1e-4), name=f'{name_prefix}_conv1b')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1b')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu1b')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=f'{name_prefix}_pool1')(x)

        # Block 2: Increase filters, capture more complex features
        x = layers.Conv2D(64, (3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(1e-4), name=f'{name_prefix}_conv2a')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2a')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu2a')(x)
        x = layers.Conv2D(64, (3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(1e-4), name=f'{name_prefix}_conv2b')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2b')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu2b')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=f'{name_prefix}_pool2')(x)

        # Block 3: Further increase filters for higher-level feature abstraction
        x = layers.Conv2D(128, (3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(1e-4), name=f'{name_prefix}_conv3a')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn3a')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu3a')(x)
        x = layers.Conv2D(128, (3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(1e-4), name=f'{name_prefix}_conv3b')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn3b')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu3b')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=f'{name_prefix}_pool3')(x)

        # Block 4: Continue increasing filter depth
        x = layers.Conv2D(256, (3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(1e-4), name=f'{name_prefix}_conv4a')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn4a')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu4a')(x)
        x = layers.Conv2D(256, (3, 3), padding='same',
                          kernel_regularizer=regularizers.l2(1e-4), name=f'{name_prefix}_conv4b')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn4b')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu4b')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=f'{name_prefix}_pool4')(x)

        # Block 5: Final convolutional block in the backbone
        # Note: Filter count adjusted to 512 in the theoretical write-up for deeper abstraction,
        # but kept at 256 here based on provided code's last working state.
        # If input size is 256x256, after 5 pools, spatial dim becomes 8x8.
        x = layers.Conv2D(256, (3, 3), padding='same', name='conv5a',
                          kernel_regularizer=regularizers.l2(1e-4))(x)  # Added L2 regularization for consistency
        x = layers.BatchNormalization(name='bn5a')(x)
        x = layers.Activation('relu', name='relu5a')(x)
        x = layers.Conv2D(256, (3, 3), padding='same', name='conv5b',
                          kernel_regularizer=regularizers.l2(1e-4))(x)  # Added L2 regularization for consistency
        x = layers.BatchNormalization(name='bn5b')(x)
        x = layers.Activation('relu', name='relu5b')(x)
        x = layers.MaxPooling2D((2, 2), name='pool5')(x)

        return x

    def _build_model(self) -> models.Model:
        """
        Constructs the complete Keras model for bone age prediction.

        The model integrates a CNN backbone for feature extraction,
        a Spatial Attention layer, and a multi-layered regression head.

        Returns:
            tf.keras.Model: The compiled Keras Model instance.
        """
        # Define the input layer for preprocessed images.
        # The input shape is (height, width, channels), where channels=1 for grayscale.
        prep_input = layers.Input(shape=(*self.img_size, 1), name='prep_input')

        # Create the CNN branch for feature extraction from the preprocessed input.
        prep_features = self._create_cnn_branch(prep_input, 'prep')

        # Apply the custom Spatial Attention mechanism to the extracted features.
        # This layer selectively re-weights spatial regions, focusing on diagnostically
        # relevant areas of the radiograph.
        attended_prep_features = SpatialAttention(name='attention_prep')(prep_features)

        # Flatten the attended features to prepare for the fully connected layers.
        x = layers.Flatten(name='flatten_features')(attended_prep_features)

        # --- Regression Head: Fully Connected Layers for Age Prediction ---
        # Dense Layer 1: Processes the flattened features.
        # Followed by Batch Normalization and Dropout for regularization.
        x = layers.Dense(512, activation='relu', name='dense1',
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization(name='bn_dense1')(x)
        x = layers.Dropout(0.4, name='dropout1')(x)

        # Dense Layer 2: Further refines the features.
        # Includes Batch Normalization and Dropout.
        x = layers.Dense(256, activation='relu', name='dense2',
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization(name='bn_dense2')(x)
        x = layers.Dropout(0.4, name='dropout2')(x)

        # Dense Layer 3: Adds more complexity to the mapping.
        # Also includes Batch Normalization and Dropout.
        x = layers.Dense(128, activation='relu', name='dense3',
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization(name='bn_dense3')(x)
        x = layers.Dropout(0.3, name='dropout3')(x)

        # Final Output Layer: Predicts the bone age in months.
        # 'linear' activation allows for any real value output.
        # A subsequent 'relu' activation is applied to ensure predictions are non-negative.
        output_linear = layers.Dense(1, name='age_output_linear',
                                     kernel_regularizer=regularizers.l2(1e-4))(x)  # Added L2 for consistency
        # Force predictions to be non-negative (bone age cannot be < 0)
        output = layers.Activation('relu', name='age_output_relu')(output_linear)

        # Create the Keras Model instance, defining its inputs and outputs.
        model = models.Model(inputs=prep_input, outputs=output, name='AgePredictionModel')
        return model

    def compile_model(self, learning_rate: float = 0.0005):
        """
        Compiles the Keras model with a specified optimizer, loss function, and metrics.

        Args:
            learning_rate (float, optional): The initial learning rate for the Adam optimizer.
                                             Defaults to 0.0005.
        """
        # Use Adam optimizer for efficient training with adaptive learning rates.
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Compile the model specifying Mean Absolute Error (MAE) as the loss function
        # (directly interpretable as error in months) and also track it as a metric.
        self.model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])