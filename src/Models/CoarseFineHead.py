from keras.src.saving import register_keras_serializable
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2


# ----------------------------
# 3) Head coarse-to-fine:
#    - ordinal regression sugli anni (CORAL)
#    - regressione mesi residui
# ----------------------------
@register_keras_serializable(package="Custom")
class CoarseFineHead(Model):
    def __init__(self, max_years=20, l2_reg=0.001, **kwargs):
        super().__init__(**kwargs)
        self.max_years = max_years
        self.l2_reg = l2_reg

        # Layer condiviso più semplice
        self.shared = Dense(32, activation='relu', kernel_regularizer=l2(l2_reg))

        # Head A: ordinal anni (più stabile senza layer intermedi)
        self.ordinal = Dense(max_years, activation='sigmoid', name='ordinal_years',
                             kernel_regularizer=l2(l2_reg))

        # Head B: regressione mesi residui
        self.residual = Dense(1, activation='linear', name='residual_months',
                              kernel_regularizer=l2(l2_reg))

    def call(self, x):
        shared_feat = self.shared(x)
        return self.ordinal(shared_feat), self.residual(shared_feat)

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_years': self.max_years,
            'l2_reg': self.l2_reg
        })
        return config