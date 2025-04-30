import os
from tensorflow.keras.models import load_model
import tensorflow as tf

from src.Models import GradientReversal, GenderAdversarialHead, CoarseFineHead
from src.Models.CNNbackbone import RadiographBackbone
from src.Models.GradientReversal import GradientReversal
from src.Models.RadiomicMLP import RadiomicsMLP
from src.loss.CoralLoss import coral_loss
from src.loss.YearLoss import months_mae, years_exact_acc


def load_saved_model(model_path):
    """
    Carica un modello AgeEstimator dai pesi salvati
    """
    # Carica le configurazioni del modello
    import json
    try:
        with open(os.path.join(model_path, "components_config.json"), "r") as json_file:
            config = json.load(json_file)
    except FileNotFoundError:
        print("File di configurazione non trovato. Usando valori predefiniti.")
        config = {
            'input_shape': (128, 128, 1),
            'radiomics_dim': 4,
            'max_years': 20
        }

    # Crea un nuovo modello con la stessa architettura
    model = AgeEstimator(
        input_shape=config.get('input_shape', (128, 128, 1)),
        radiomics_dim=config.get('radiomics_dim', 4),
        max_years=config.get('max_years', 20)
    )

    # Costruisci il modello per inizializzare i pesi
    model_graph = model.build_graph()

    # Compila il modello
    lambda_adversarial = -0.1
    weight_months = 0.5

    model_graph.compile(
        optimizer='adam',
        loss={
            "ordinal_output": coral_loss,
            "month_output": 'mae',
            "gender_out": 'binary_crossentropy'
        },
        loss_weights={
            "ordinal_output": 1.0,
            "month_output": weight_months,
            "gender_out": lambda_adversarial
        },
        metrics={
            'coarse_fine_head': [months_mae],
            'gender_adversarial_head': ['accuracy']
        }
    )

    # Carica i pesi salvati
    try:
        model_graph.load_weights(os.path.join(model_path, "model_weights.h5"))
        print("Pesi caricati con successo.")
    except Exception as e:
        print(f"Errore nel caricare i pesi: {e}")
        return None

    return model_graph


def try_load_directly(model_path):
    """
    Tenta di caricare il modello completo direttamente
    """
    custom_objects = {
        "RadiographBackbone": RadiographBackbone,
        "RadiomicsMLP": RadiomicsMLP,
        "GradientReversal": GradientReversal,
        "GenderAdversarialHead": GenderAdversarialHead,
        "CoarseFineHead": CoarseFineHead,
        "coral_loss": coral_loss,
        "months_mae": months_mae,
        "AgeEstimator": AgeEstimator
    }

    try:
        model = load_model(os.path.join(model_path, "model_full.h5"), custom_objects=custom_objects)
        print("Modello caricato direttamente con successo.")
        return model
    except Exception as e:
        print(f"Caricamento diretto fallito: {e}")
        print("Tentativo di caricamento dai pesi...")
        return None