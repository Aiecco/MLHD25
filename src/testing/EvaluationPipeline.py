import os

import numpy as np

from src.Models.AttentionLayer import SpatialAttention
from src.plot.PlotEval import plot_eval
from src.testing.evaluate import evaluate_saved_model
from keras import models


def evaluation_pipeline(model_save_path, test_path, label_path):
    # --- Valutazione del modello salvato sul Test Set e plotting ---
    print("\n--- Valutazione del modello salvato e generazione grafici ---")
    if os.path.exists(model_save_path):
        evaluation_results, true_months_list, pred_months_list = evaluate_saved_model(model_save_path, test_path, label_path)
        loaded_model = models.load_model(model_save_path, custom_objects={'SpatialAttention': SpatialAttention})
        if evaluation_results is not None:
            # Trova il MAE dal risultato della valutazione
            mae_index = loaded_model.metrics_names.index('mae') if 'mae' in loaded_model.metrics_names else 1
            mae_months = evaluation_results[mae_index] if mae_index is not None else np.nan

            # Calcola gli errori assoluti per il plot
            errors_list = np.abs(true_months_list - pred_months_list)

            # Chiama la funzione di plotting
            plot_eval(errors_list, mae_months, true_months_list, pred_months_list)
            print("Grafici di valutazione generati e salvati come 'age_prediction_analysis.png'.")
        else:
            print("\nImpossibile valutare il modello salvato a causa di un errore nel caricamento.")
    else:
        print(f"Errore: Il modello non è stato trovato al percorso specificato: {model_save_path}")