from keras.src.layers import Activation, Dropout
from keras.src.saving import register_keras_serializable
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras import backend as K  # Importa il backend di Keras
from tensorflow.keras.regularizers import l2


@register_keras_serializable(package="Custom")
class RadiographBackbone(Model):
    def __init__(self, filters=[16, 32, 64], kernel_size=3, dropout_rate=0.2, l2_reg=0.001, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        # Stack convoluzionale semplice ma stabile
        self.convs = []
        for f in filters:
            # Conv + BN + ReLU
            self.convs.append(Conv2D(f, kernel_size, padding='same', activation=None,
                                     kernel_regularizer=l2(l2_reg)))
            self.convs.append(BatchNormalization())
            self.convs.append(Activation('relu'))
            # Pooling
            self.convs.append(MaxPooling2D(2))
            # Dropout spaziale
            self.convs.append(Dropout(dropout_rate))

        # Output pooling
        self.global_pool = GlobalAveragePooling2D()

    def call(self, x, training=False):
        for layer in self.convs:
            if isinstance(layer, (BatchNormalization, Dropout)):
                x = layer(x, training=training)
            else:
                x = layer(x)

        return self.global_pool(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg
        })
        return config
