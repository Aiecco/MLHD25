import os
import json
import tensorflow as tf
from tensorflow.keras.models import load_model

from src.Models import GradientReversal, GenderAdversarialHead, CoarseFineHead
from src.Models.CNNbackbone import RadiographBackbone
from src.Models.GradientReversal import GradientReversal
from src.Models.OrdinalRegressor import AgeEstimator
from src.Models.RadiomicMLP import RadiomicsMLP
from src.loss.YearLoss import months_mae


def load_saved_model(model_path, loss_weight_month, loss_weight_gender):
    """
    Carica un modello AgeEstimator completo. Priorità:
    1. Caricamento da file .keras (salvataggio completo)
    2. Fallback: costruzione del modello + caricamento pesi separati
    """
    # === 1. TENTATIVO DI CARICAMENTO DIRETTO (.keras) ===
    model_file = os.path.join(model_path, "age_estimator.keras")
    if os.path.exists(model_file):
        try:
            print(f"Tento il caricamento diretto da {model_file}...")
            model = load_model(model_file, custom_objects={
                "RadiographBackbone": RadiographBackbone,
                "RadiomicsMLP": RadiomicsMLP,
                "GradientReversal": GradientReversal,
                "GenderAdversarialHead": GenderAdversarialHead,
                "CoarseFineHead": CoarseFineHead,
                "AgeEstimator": AgeEstimator,
                "months_mae": months_mae
            })
            print("✅ Modello caricato direttamente con successo.")
            return model
        except Exception as e:
            print(f"⚠️ Caricamento diretto fallito: {e}")

    # === 2. Fallback: ricostruzione + caricamento pesi ===
    print("🔁 Fallback: ricostruisco il modello da zero...")

    # Config default in caso manchi il JSON
    config_path = os.path.join(model_path, "components_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as json_file:
            config = json.load(json_file)
    else:
        print("⚠️ File di configurazione non trovato. Uso config di default.")
        config = {
            'input_shape': (128, 128, 1),
            'radiomics_dim': 38,
            'max_years': 20
        }

    # Istanzia modello subclassed
    model_wrapper = AgeEstimator(
        input_shape=config['input_shape'],
        radiomics_dim=config['radiomics_dim'],
        max_years=config['max_years']
    )

    # Costruisce il modello funzionale
    model_graph = model_wrapper.build_graph()

    # Compilazione (come nel training)
    model_graph.compile(
        optimizer='adam',
        loss={
            "month_output": 'mae',
            "gender_out": 'binary_crossentropy'
        },
        loss_weights={
            "month_output": loss_weight_month,
            "gender_out": loss_weight_gender
        },
        metrics={
            'coarse_fine_head': [months_mae],
            'gender_adversarial_head': ['accuracy']
        }
    )

    # Caricamento dei pesi
    weights_path = os.path.join(model_path, "age_estimator.weights.h5")
    try:
        model_graph.load_weights(weights_path)
        print("✅ Pesi caricati con successo.")
        return model_graph
    except Exception as e:
        print(f"❌ Errore nel caricare i pesi: {e}")
        return None