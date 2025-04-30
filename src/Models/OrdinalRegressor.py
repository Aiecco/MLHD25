from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate

from src.Models.CNNbackbone import RadiographBackbone
from src.Models.CoarseFineHead import CoarseFineHead
from src.Models.GenderAdversarialHead import GenderAdversarialHead
from src.Models.GradientReversal import GradientReversal
from src.Models.RadiomicMLP import RadiomicsMLP


class AgeEstimator(Model):
    def __init__(self,
                 input_shape=(128, 128, 1),
                 radiomics_dim=50,
                 max_years=20,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.radiomics_dim = radiomics_dim
        self.max_years = max_years

        self.backbone = RadiographBackbone()
        self.rad_mlp = RadiomicsMLP()
        self.fusion_dense_1 = Dense(128, activation='relu')
        self.fusion_dense_2 = Dense(128, activation='relu')
        self.dropout = Dropout(0.3)
        self.head = CoarseFineHead(max_years=max_years)
        self.grl_head = GradientReversal(hp_lambda=0.5)
        self.adv_head = GenderAdversarialHead()

    def call(self, inputs, training=False):
        if isinstance(inputs, dict):
            img = inputs.get('radiograph', inputs.get('img', None))
            rad = inputs.get('radiomics', inputs.get('rad', None))
            gender = inputs.get('gender', None)
        elif isinstance(inputs, (list, tuple)) and len(inputs) >= 2:
            img = inputs[0]
            rad = inputs[1]
            gender = inputs[2] if len(inputs) > 2 else None
        else:
            raise ValueError("Input format not recognized")

        feat_img = self.backbone(img, training=training)
        feat_rad = self.rad_mlp(rad, training=training)

        x = Concatenate()([feat_img, feat_rad])
        x = self.fusion_dense_1(x)
        x = self.dropout(x, training=training)

        h = self.fusion_dense_2(x)
        h = self.dropout(h, training=training)

        ord_logits, month_out = self.head(h)

        h_rev = self.grl_head(h)
        gender_pred = self.adv_head(h_rev)

        return {'ordinal_output': ord_logits, 'month_output': month_out, 'gender_out': gender_pred}

    def build_graph(self):
        img_input = Input(shape=self.input_shape, name='radiograph')
        rad_input = Input(shape=(self.radiomics_dim,), name='radiomics')

        outputs = self.call({'radiograph': img_input, 'radiomics': rad_input}, training=False)

        return Model(inputs=[img_input, rad_input], outputs=outputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'radiomics_dim': self.radiomics_dim,
            'max_years': self.max_years
        })
        return config