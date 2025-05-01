from keras.src.layers import Dense
from keras.src.saving import register_keras_serializable
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l2


@register_keras_serializable(package="Custom")
class GenderAdversarialHead(Model):
    def __init__(self, l2_reg=0.001, **kwargs):
        super().__init__(**kwargs)
        # Semplificato a un solo layer
        self.out = Dense(1, activation='sigmoid', name='gender_pred',
                         kernel_regularizer=l2(l2_reg))

    def call(self, x, training=False):
        return self.out(x)

    def get_config(self):
        config = super().get_config()
        return config

