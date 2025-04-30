from tensorflow.keras import Model
from tensorflow.keras.layers import Dense


# ----------------------------
# 3) Head coarse-to-fine:
#    - ordinal regression sugli anni (CORAL)
#    - regressione mesi residui
# ----------------------------
class CoarseFineHead(Model):
    def __init__(self, max_years=20, **kwargs):
        super().__init__(**kwargs)
        self.max_years = max_years
        # Head A: ordinal anni
        self.ordinal = Dense(max_years, activation='sigmoid', name='ordinal_years')
        # Head B: regressione mesi residui
        self.residual = Dense(1, activation='linear', name='residual_months')

    def call(self, x):
        return self.ordinal(x), self.residual(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_years': self.max_years
        })
        return config