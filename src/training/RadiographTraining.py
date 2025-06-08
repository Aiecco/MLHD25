import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

def train_model(model, train_dataset, epochs=50, validation_dataset=None,
                model_save_path='best_age_prediction_model.keras'):
    """
    Addestra il modello fornito, gestendo la pipeline di training con callback.

    Args:
        model (tf.keras.Model): Il modello Keras già compilato.
        train_dataset (tf.data.Dataset): Il dataset per l'addestramento,
                                         che emette ((raw_img, prep_img, extr_img), age_months).
        epochs (int): Numero di epoche di addestramento.
        validation_dataset (tf.data.Dataset, optional): Il dataset per la validazione.
                                                        Deve avere la stessa struttura di train_dataset.
                                                        Defaults to None.
        model_save_path (str): Percorso dove salvare il miglior modello durante l'addestramento.

    Returns:
        tf.keras.callbacks.History: L'oggetto History con la cronologia dell'addestramento.
        tf.keras.Model: Il modello addestrato (con i pesi migliori se EarlyStopping è attivo).
    """

    print(f"\nAvvio dell'addestramento del modello per {epochs} epoche...")

    # Callback per un controllo migliore dell'addestramento
    callbacks = [
        # Salva il modello migliore basato sulla MAE di validazione
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_mae' if validation_dataset else 'mae',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        # Interrompe l'addestramento se la MAE di validazione non migliora
        EarlyStopping(
            monitor='val_mae' if validation_dataset else 'mae',
            patience=10, # Numero di epoche senza miglioramento dopo le quali l'addestramento verrà interrotto.
            mode='min',
            verbose=1,
            restore_best_weights=True # Ripristina i pesi del modello dall'epoca con il miglior valore monitorato.
        ),
        # Riduce il learning rate quando una metrica smette di migliorare
        ReduceLROnPlateau(
            monitor='val_mae' if validation_dataset else 'mae',
            factor=0.5, # Fattore di riduzione del learning rate (nuovo_lr = lr * factor).
            patience=5, # Numero di epoche senza miglioramento dopo le quali il learning rate verrà ridotto.
            min_lr=1e-6, # Limite inferiore per il learning rate.
            mode='min',
            verbose=1
        )
    ]

    # Addestra il modello
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=validation_dataset,
        callbacks=callbacks
    )
    print("\nAddestramento del modello completato.")
    return model, history