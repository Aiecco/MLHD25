from keras.src.saving import register_keras_serializable
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout

# ----------------------------
# 2) MLP per vettore radiomico
# ---------------------------
@register_keras_serializable(package="Custom")
class RadiomicsMLP(Model):
    def __init__(self, hidden_units=[64, 32], dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout
        self.net = []
        for u in hidden_units:
            self.net.append(Dense(u, activation='relu'))
            self.net.append(Dropout(dropout))

    def call(self, x, training=False):
        for layer in self.net:
            # Dropout riceve training flag
            if isinstance(layer, Dropout):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_units': self.hidden_units,
            'dropout': self.dropout_rate
        })
        return config