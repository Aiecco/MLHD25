# ----------------------------
# 1) Backbone CNN per radiografie
# ----------------------------
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D


class RadiographBackbone(Model):
    def __init__(self, filters=[32, 64, 128], kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.convs = []
        for f in filters:
            self.convs.append(Conv2D(f, kernel_size, padding='same', activation='relu'))
            self.convs.append(BatchNormalization())
            self.convs.append(MaxPooling2D(2))
        self.global_pool = GlobalAveragePooling2D()

    def call(self, x, training=False):
        for layer in self.convs:
            # batchnorm layer gestisce training flag, conv/ReLU/Pool no
            if isinstance(layer, BatchNormalization):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return self.global_pool(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size
        })
        return config