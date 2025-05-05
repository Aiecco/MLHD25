from keras.src import layers, regularizers
from keras.src.saving import register_keras_serializable
from tensorflow.keras.regularizers import l2


@register_keras_serializable(package="Custom")
class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size=2, strides=1, use_pooling=True, pool_type='avg', l2_reg=0.001):
        super(ResidualBlock, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.l2_reg = l2_reg
        self.use_pooling = use_pooling
        self.pool_type = pool_type

        self.conv1 = layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                                   kernel_regularizer=l2(l2_reg))
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same',
                                   kernel_regularizer=l2(l2_reg))
        self.bn2 = layers.BatchNormalization()

        self.shortcut = None
        if strides != 1:
            self.shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same',
                                          kernel_regularizer=l2(l2_reg))

        self.relu_out = layers.ReLU()
        self.use_pooling = use_pooling
        if use_pooling:
            self.pool = layers.AveragePooling2D(2) if pool_type == 'avg' else layers.MaxPooling2D(2)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        shortcut = self.shortcut(inputs) if self.shortcut is not None else inputs
        x = layers.add([x, shortcut])
        x = self.relu_out(x)

        if self.use_pooling:
            x = self.pool(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'use_pooling': self.use_pooling,
            'pool_type': self.pool_type,
            'l2_reg': self.l2_reg
        })
        return config