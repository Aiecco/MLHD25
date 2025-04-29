import tensorflow as tf

# Define a simple custom MAE metric class that extends directly from MeanAbsoluteError
class CustomMAE(tf.keras.metrics.MeanAbsoluteError):
    def __init__(self, name='mae', **kwargs):
        super(CustomMAE, self).__init__(name=name, **kwargs)