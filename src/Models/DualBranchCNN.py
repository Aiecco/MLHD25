import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model, Input

class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, use_pooling=True, pool_type='avg', l2_reg=0.001):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(
            filters, kernel_size, strides=strides, padding='same',
            kernel_regularizer=regularizers.l2(l2_reg)
        )
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        
        self.conv2 = layers.Conv2D(
            filters, kernel_size, padding='same',
            kernel_regularizer=regularizers.l2(l2_reg)
        )
        self.bn2 = layers.BatchNormalization()
        
        # Projection shortcut if dimensions change
        self.shortcut = None
        if strides != 1:
            self.shortcut = layers.Conv2D(
                filters, 1, strides=strides, padding='same',
                kernel_regularizer=regularizers.l2(l2_reg)
            )
        
        self.relu_out = layers.ReLU()
        self.use_pooling = use_pooling
        if use_pooling:
            if pool_type == 'avg':
                self.pool = layers.AveragePooling2D(2)
            else:
                self.pool = layers.MaxPooling2D(2)
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        if self.shortcut is not None:
            shortcut = self.shortcut(inputs)
        else:
            shortcut = inputs
        
        x = layers.add([x, shortcut])
        x = self.relu_out(x)
        
        if self.use_pooling:
            x = self.pool(x)
        
        return x

def create_dual_branch_cnn(input_channels=1, img_size=(128, 128), gender_dim=1, l2_reg=0.001):
    """
    Create a DualBranchCNN model using the Functional API
    This makes it easier to use with model.fit()
    """
    # Define inputs
    pooled_input = Input(shape=(img_size[0], img_size[1], input_channels), name='pooled_input')
    heatmap_input = Input(shape=(img_size[0], img_size[1], input_channels), name='heatmap_input')
    gender_input = Input(shape=(gender_dim,), name='gender_input', dtype='float32')
    
    # --- Branch 1: Immagini Pooled con Residual Connections ---
    x1 = layers.Conv2D(
        64, kernel_size=7, strides=2, padding='same', activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(pooled_input)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling2D(3, strides=2, padding='same')(x1)
    
    # Residual blocks for branch 1
    x1 = ResidualBlock(64, use_pooling=False, l2_reg=l2_reg)(x1)
    x1 = ResidualBlock(128, strides=2, use_pooling=False, l2_reg=l2_reg)(x1)
    x1 = ResidualBlock(256, strides=2, use_pooling=False, l2_reg=l2_reg)(x1)
    x1 = ResidualBlock(512, strides=2, use_pooling=False, l2_reg=l2_reg)(x1)
    x1 = layers.GlobalAveragePooling2D()(x1)
    
    # --- Branch 2: Heatmaps con Residual Connections ---
    x2 = layers.Conv2D(
        64, kernel_size=7, strides=2, padding='same', activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(heatmap_input)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D(3, strides=2, padding='same')(x2)
    
    # Residual blocks for branch 2
    x2 = ResidualBlock(64, use_pooling=False, pool_type='max', l2_reg=l2_reg)(x2)
    x2 = ResidualBlock(128, strides=2, use_pooling=False, pool_type='max', l2_reg=l2_reg)(x2)
    x2 = ResidualBlock(256, strides=2, use_pooling=False, pool_type='max', l2_reg=l2_reg)(x2)
    x2 = ResidualBlock(512, strides=2, use_pooling=False, pool_type='max', l2_reg=l2_reg)(x2)
    x2 = layers.GlobalAveragePooling2D()(x2)
    
    # Process gender input - already float32 as specified in the Input
    # Use Lambda layer for any reshaping needed
    gender_features = layers.Reshape((gender_dim,))(gender_input)
    
    # Concatenate features - ensure they're all in the right format
    x = layers.Concatenate(axis=1)([x1, x2, gender_features])
    
    # Fully connected layers
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, name='age_output')(x)
    
    # Create model with explicit input dictionary for clearer mapping
    model = Model(
        inputs={
            'pooled_input': pooled_input,
            'heatmap_input': heatmap_input,
            'gender_input': gender_input
        },
        outputs=output
    )
    
    return model


class DualBranchCNN(Model):
    def __init__(self, input_channels=1, img_size=(128, 128), gender_dim=1, l2_reg=0.001, **kwargs):
        super(DualBranchCNN, self).__init__(**kwargs)
        
        # Store parameters for config
        self.input_channels = input_channels
        self.img_size = img_size
        self.gender_dim = gender_dim
        self.l2_reg = l2_reg
        
        # Create functional model
        self._functional_model = create_dual_branch_cnn(
            input_channels=input_channels,
            img_size=img_size,
            gender_dim=gender_dim,
            l2_reg=l2_reg
        )
        
        # Set model inputs and outputs to match the functional model
        self.inputs = self._functional_model.inputs
        self.outputs = self._functional_model.outputs

    def call(self, inputs, training=False):
        """
        Forward pass through the model
        
        Args:
            inputs: Can be either:
                - a dict with keys 'pooled_input', 'heatmap_input', 'gender_input'
                - compatible tensor inputs
            training: Whether to run in training mode
            
        Returns:
            The predicted age
        """
        return self._functional_model(inputs, training=training)

    def get_config(self):
        config = super(DualBranchCNN, self).get_config()
        config.update({
            'input_channels': self.input_channels,
            'img_size': self.img_size,
            'gender_dim': self.gender_dim,
            'l2_reg': self.l2_reg,
        })
        return config
        
    def summary(self, *args, **kwargs):
        """Forward the summary call to the functional model"""
        return self._functional_model.summary(*args, **kwargs)
        
    @classmethod
    def from_config(cls, config):
        """Create model from config"""
        return cls(**config)
        
    def save(self, filepath, *args, **kwargs):
        """Forward the save call to the functional model"""
        return self._functional_model.save(filepath, *args, **kwargs)

    def load_weights(self, filepath, *args, **kwargs):
        """Forward the load_weights call to the functional model"""
        return self._functional_model.load_weights(filepath, *args, **kwargs)
