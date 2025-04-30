import tensorflow as tf
from tensorflow.keras.layers import Layer

class GradientReversal(Layer):
    def __init__(self, hp_lambda=1.0, **kwargs):
        super().__init__(**kwargs)
        self.hp_lambda = hp_lambda

    def call(self, x):
        @tf.custom_gradient
        def _flip_grad(x):
            def grad(dy):
                return -self.hp_lambda * dy
            return x, grad
        return _flip_grad(x)
