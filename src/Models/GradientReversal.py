import tensorflow as tf
from tensorflow.keras.layers import Layer


class GradientReversal(tf.keras.layers.Layer):
    def __init__(self, hp_lambda=1.0, **kwargs):
        super().__init__(**kwargs)
        self.hp_lambda = hp_lambda

    def call(self, x, training=None):
        # Forward pass: identità
        return x

    def get_config(self):
        config = super().get_config()
        config.update({'hp_lambda': self.hp_lambda})
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

    # Sovrascrittura del comportamento del gradiente
    @tf.custom_gradient
    def grad_reverse(self, x):
        y = tf.identity(x)

        def custom_grad(dy):
            return -self.hp_lambda * dy

        return y, custom_grad