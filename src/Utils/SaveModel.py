import os


def save_model_properly(model, save_path):
    """
    Salva il modello in formati multipli per garantire la possibilità di caricamento
    """
    os.makedirs(save_path, exist_ok=True)

    # 1. Salva i pesi del modello separatamente
    model.save_weights(os.path.join(save_path, "age_estimator.weights.h5"))

    # 2. Salva le configurazioni dei componenti personalizzati
    components_config = {
        'input_shape': model.input_shape,
        'radiomics_dim': model.radiomics_dim if hasattr(model, 'radiomics_dim') else 4,
        'max_years': model.max_years if hasattr(model, 'max_years') else 20
    }

    import json
    with open(os.path.join(save_path, "components_config.json"), "w") as json_file:
        json.dump(components_config, json_file)

    # 3. Tenta di salvare il modello intero (potrebbe non funzionare al caricamento)
    try:
        model.save(os.path.join(save_path, "age_estimator.h5"))
        print("Modello completo salvato con successo.")
    except Exception as e:
        print(f"Errore nel salvare il modello completo: {e}")
        print("Si consiglia di caricare il modello utilizzando i pesi salvati separatamente.")

    print(f"Modello salvato con successo in {save_path}")