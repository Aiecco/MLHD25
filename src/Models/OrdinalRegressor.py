from keras.src.saving import register_keras_serializable
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from tensorflow.keras.regularizers import l2
import tensorflow as tf

from src.Models.CNNbackbone import RadiographBackbone
from src.Models.CoarseFineHead import CoarseFineHead
from src.Models.GenderAdversarialHead import GenderAdversarialHead
from src.Models.GradientReversal import GradientReversal
from src.Models.RadiomicMLP import RadiomicsMLP


@register_keras_serializable(package="Custom")
class AgeEstimator(Model):
    def __init__(self,
                 input_shape=(128, 128, 1),
                 radiomics_dim=38,
                 max_years=20,
                 l2_reg=0.001,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.radiomics_dim = radiomics_dim
        self.max_years = max_years
        self.l2_reg = l2_reg

        # Componenti principali semplificati
        self.backbone = RadiographBackbone(filters=[16, 32, 64], l2_reg=l2_reg)
        self.rad_mlp = RadiomicsMLP(hidden_units=[32, 16], l2_reg=l2_reg)

        # Fusion semplificata
        self.fusion_dense = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))
        self.dropout = Dropout(0.3)

        # Heads
        self.head = CoarseFineHead(max_years=max_years, l2_reg=l2_reg)
        self.grl_head = GradientReversal(hp_lambda=0.1)  # Lambda molto più basso
        self.adv_head = GenderAdversarialHead(l2_reg=l2_reg)

    def call(self, inputs, training=False):
        # Estrai input
        if isinstance(inputs, dict):
            img = inputs.get('radiograph', inputs.get('img', None))
            rad = inputs.get('radiomics', inputs.get('rad', None))
        elif isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
            img = inputs[0]
            rad = inputs[1]
        else:
            raise ValueError("Input format not recognized")

        # Feature extraction
        feat_img = self.backbone(img, training=training)
        feat_rad = self.rad_mlp(rad, training=training)

        # Semplice concatenazione
        x = Concatenate()([feat_img, feat_rad])

        # Fusion network semplificata
        h = self.fusion_dense(x)
        h = self.dropout(h, training=training)

        # Outputs
        age_pred_months = self.head(h)

        # Task avversariale con GRL stabile
        h_rev = self.grl_head(h)
        gender_pred = self.adv_head(h_rev)

        return {
            'month_output': age_pred_months,
            'gender_out': gender_pred
        }

    def build_graph(self):
        # Gestione sicura degli input shape
        current_input_shape = tuple(map(int, self.input_shape))
        current_radiomics_dim = int(self.radiomics_dim)

        # Definizione input
        img_input = Input(shape=current_input_shape, name='radiograph')
        rad_input = Input(shape=(current_radiomics_dim,), name='radiomics')

        # Forward pass
        outputs = self.call({'radiograph': img_input, 'radiomics': rad_input}, training=False)

        # Modello funzionale
        return Model(inputs=[img_input, rad_input], outputs=outputs, name=self.name or "AgeEstimatorFunctional")

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'radiomics_dim': self.radiomics_dim,
            'max_years': self.max_years,
            'l2_reg': self.l2_reg
        })
        return config
