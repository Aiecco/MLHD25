import tensorflow as tf
from keras.src.saving import register_keras_serializable
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K

@register_keras_serializable(package="Custom")
class GradientReversal(tf.keras.layers.Layer):
    def __init__(self, hp_lambda=0.1, **kwargs):
        super().__init__(**kwargs)
        self.hp_lambda = hp_lambda

    def call(self, x, training=False):
        @tf.custom_gradient
        def _reverse_grad(x):
            def grad(dy):
                return -self.hp_lambda * dy
            return x, grad

        # se training=True inverto il gradiente, altrimenti identity
        return _reverse_grad(x) if training else x

    def get_config(self):
        config = super().get_config()
        config.update({'hp_lambda': self.hp_lambda})
        return config