# --- Funzione di Valutazione per Modello Salvato ---
import numpy as np
import tensorflow as tf
from src.Models.AttentionLayer import SpatialAttention
from keras import models

from src.dataset.RadiographDataset import RadiographDatasetBuilder


# --- Funzione di Valutazione per Modello Salvato ---
def evaluate_saved_model(model_path, test_dataset_path, label_test_dataset_path, img_sizes=128):
    """
    Carica un modello Keras salvato, lo valuta su un dataset di test e raccoglie
    le predizioni e le etichette vere.

    Args:
        model_path (str): Il percorso del file del modello Keras salvato (.keras, .h5, o SavedModel directory).
        test_dataset (tf.data.Dataset): Il dataset su cui valutare il modello.
                                        Dovrebbe avere la stessa struttura prevista dal modello per l'input.

    Returns:
        tuple: (results, true_months, pred_months) dove:
               - results: I risultati della valutazione (es. loss, metrics).
               - true_months: Lista delle etichette vere in mesi.
               - pred_months: Lista delle predizioni del modello in mesi.
               Returns (None, None, None) se il modello non può essere caricato.
    """
    # Builder per il Test Set
    builder_test = RadiographDatasetBuilder(
        base_dir=test_dataset_path,
        label_csv=label_test_dataset_path,
        img_size=(img_sizes, img_sizes),
        batch_size=1
    )
    test_dataset = builder_test.build(
        train=False)  # Non shufflare il test set. Batch size 1 per predizioni individuali.

    print(f"\nCaricamento del modello da: {model_path}")
    try:
        # Carica il modello. custom_objects è necessario per il layer SpatialAttention.
        loaded_model = models.load_model(model_path, custom_objects={'SpatialAttention': SpatialAttention})
        print("Modello caricato con successo.")
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
        return None, None, None

    print("\nValutazione e raccolta predizioni sul Test Set:")

    true_months = []
    pred_months = []

    # Iteriamo sul dataset per raccogliere le vere etichette e fare le predizioni
    # Usiamo .unbatch() per processare i singoli esempi
    for inputs, labels in test_dataset.unbatch():
        # Aggiungiamo una dimensione batch per la predizione di un singolo elemento
        input_batch = tf.expand_dims(inputs, axis=0)

        prediction = loaded_model.predict(input_batch, verbose=0)[0][0]  # Prendi il valore scalare
        true_months.append(labels.numpy())
        pred_months.append(prediction)

    # Convertiamo le liste in array NumPy per i calcoli successivi
    true_months = np.array(true_months)
    pred_months = np.array(pred_months)

    # Eseguiamo la valutazione formale del modello
    results = loaded_model.evaluate(test_dataset, verbose=0)

    print("Risultati della valutazione:")
    for name, value in zip(loaded_model.metrics_names, results):
        print(f"{name}: {value:.4f}")

    return results, true_months, pred_months