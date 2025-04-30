from tensorflow.keras import layers, Model


class GenderAdversarialHead(Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense1 = layers.Dense(32, activation='relu')
        self.out = layers.Dense(1, activation='sigmoid', name='gender_pred')

    def call(self, x, training=False):
        x = self.dense1(x)
        return self.out(x)

    def get_config(self):
        config = super().get_config()
        return config
