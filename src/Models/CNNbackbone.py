from keras.src.layers import Activation, Dropout, Dense
from keras.src.saving import register_keras_serializable
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras import backend as K  # Importa il backend di Keras
from tensorflow.keras.regularizers import l2

from src.Models.ResidualBlock import ResidualBlock


@register_keras_serializable(package="Custom")
class RadiographBackbone(Model):
    def __init__(self, kernel_size=2, dropout_rate=0.2, l2_reg=0.001, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        self.conv = Conv2D(32, 7, strides=2, padding='same', activation='relu',
                    kernel_regularizer=l2(self.l2_reg))

        self.residual1 = ResidualBlock(filters=32, strides=1, use_pooling=False, l2_reg=self.l2_reg)
        self.residual2 = ResidualBlock(filters=64, strides=2, use_pooling=False, l2_reg=self.l2_reg)
        self.residual3 = ResidualBlock(filters=128, strides=2, use_pooling=False, l2_reg=self.l2_reg)
        self.residual4 = ResidualBlock(filters=256, strides=2, use_pooling=False, l2_reg=self.l2_reg)

        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()
        self.bn3 = BatchNormalization()
        self.bn4 = BatchNormalization()
        self.globpool = GlobalAveragePooling2D()
        self.maxpool = MaxPooling2D(3, strides=2, padding='same')
        self.dense = Dense(16, activation='relu')

    def call(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.residual1(x)
        x = self.bn2(x)
        x = self.residual2(x)
        x = self.bn3(x)
        x = self.residual3(x)
        x = self.bn4(x)
        x = self.residual4(x)

        x = self.globpool(x)  # Applica davvero il pooling
        x = self.dense(x)  # Riduci a 16 feature
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg
        })
        return config
