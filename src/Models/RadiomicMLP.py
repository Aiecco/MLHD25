from keras.src.layers import BatchNormalization
from keras.src.saving import register_keras_serializable
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2


# ----------------------------
# 2) MLP per vettore radiomico
# ---------------------------
@register_keras_serializable(package="Custom")
class RadiomicsMLP(Model):
    def __init__(self, hidden_units=[32, 16], dropout=0.2, l2_reg=0.001, **kwargs):
        super().__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout
        self.l2_reg = l2_reg

        # Stack più semplice ma stabile
        self.net = []
        for u in hidden_units:
            self.net.append(Dense(u, activation='relu', kernel_regularizer=l2(l2_reg)))
            self.net.append(BatchNormalization())
            self.net.append(Dropout(dropout))

    def call(self, x, training=False):
        for layer in self.net:
            if isinstance(layer, (BatchNormalization, Dropout)):
                x = layer(x, training=training)
            else:
                x = layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'hidden_units': self.hidden_units,
            'dropout': self.dropout_rate,
            'l2_reg': self.l2_reg
        })
        return config