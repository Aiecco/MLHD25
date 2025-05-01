from keras.src.saving import register_keras_serializable
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Concatenate

from src.Models.CNNbackbone import RadiographBackbone
from src.Models.CoarseFineHead import CoarseFineHead
from src.Models.GenderAdversarialHead import GenderAdversarialHead
from src.Models.GradientReversal import GradientReversal
from src.Models.RadiomicMLP import RadiomicsMLP

@register_keras_serializable(package="Custom")
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

        print(f"DEBUG: In build_graph - self.input_shape: {self.input_shape}, Tipo: {type(self.input_shape)}")
        print(f"DEBUG: In build_graph - self.radiomics_dim: {self.radiomics_dim}, Tipo: {type(self.radiomics_dim)}")

        # Assicura che input_shape sia una tupla di interi
        try:
            # Converte ogni elemento della sequenza in intero e crea una tupla
            current_input_shape = tuple(map(int, self.input_shape))
        except (TypeError, ValueError) as e:
            print(f"ERRORE: Impossibile convertire self.input_shape {self.input_shape} in tupla di interi: {e}")
            # Gestisci l'errore o usa un default sicuro
            current_input_shape = (128, 128, 1) # Fallback a un valore noto

        # Assicura che radiomics_dim sia un intero
        try:
            current_radiomics_dim = int(self.radiomics_dim)
        except (TypeError, ValueError) as e:
            print(f"ERRORE: Impossibile convertire self.radiomics_dim {self.radiomics_dim} in intero: {e}")
            current_radiomics_dim = 4 # Fallback a un valore noto

        print(f"DEBUG: In build_graph - Usando shape: {current_input_shape}, radiomics: {current_radiomics_dim}")
        # --- Fine Modifica ---

        # Usa le variabili convertite
        img_input = Input(shape=current_input_shape, name='radiograph')
        rad_input = Input(shape=(current_radiomics_dim,), name='radiomics') # Passa direttamente la tupla shape

        # Assicurati che 'call' funzioni con dizionari o tuple/liste
        # Qui usiamo un dizionario come nel tuo codice originale
        outputs = self.call({'radiograph': img_input, 'radiomics': rad_input}, training=False)

        # Crea il modello funzionale
        functional_model = Model(inputs=[img_input, rad_input], outputs=outputs, name=self.name or "AgeEstimatorFunctional")
        return functional_model # Restituisci il modello funzionale creato

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_shape': self.input_shape,
            'radiomics_dim': self.radiomics_dim,
            'max_years': self.max_years
        })
        return config