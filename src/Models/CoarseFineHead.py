from keras.src.saving import register_keras_serializable
import tensorflow as tf
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
        ord_logits = self.ordinal(shared_feat)  # [batch, max_years]
        resid_pred = self.residual(shared_feat)  # [batch,1]

        # 1) passare da logit ordinali a probabilità
        probs = ord_logits  # se hai sigmoid integrato
        # 2) contare quante soglie superi → anni predetti
        years_pred = tf.reduce_sum(tf.cast(probs > 0.5, tf.float32), axis=1, keepdims=True)  # [batch,1]
        # 3) età in mesi totale
        age_pred_mths = years_pred * 12.0 + resid_pred  # [batch,1] + [batch,1] = [batch,1]

        return age_pred_mths

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_years': self.max_years,
            'l2_reg': self.l2_reg
        })
        return config