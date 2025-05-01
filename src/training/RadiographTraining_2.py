import numpy as np
from keras.src.optimizers import Adam

import tensorflow as tf

from src.loss.CoralLoss import combined_loss
from src.loss.YearLoss import months_mae


def train_model_with_monitoring(model, dataset, batch_size=64, epochs=20, validation_data=None):
    train_ds = (
        dataset
        .map(
            lambda features, labels: (
                (features[0], features[1]),  # img, radiomics, gender_input
                (labels[0], labels[1], labels[2])  # age_year, age_month, gender
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .shuffle(100)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 2) Se ho un validation_data, applico lo stesso preprocess
    if validation_data is not None:
        val_ds = (
            validation_data
            .map(
                lambda features, labels: (
                    (features[0], features[1]),
                    (labels[0], labels[1], labels[2]),
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        val_ds = None

    # Optimizer con clip norm
    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)

    # Compile
    model.compile(
        optimizer=optimizer,
        loss=combined_loss,
        metrics=[months_mae, 'accuracy']
    )

    # Custom callback per monitorare i valori
    class MonitorValuesCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            for k, v in logs.items():
                if abs(v) > 1e6:  # Controlla valori anomali
                    print(f"⚠️ ATTENZIONE: Valore anomalo {k} = {v}")

            # Salva alcuni esempi di predizioni
            batch = next(iter(val_ds))
            X, y = batch
            preds = self.model.predict(X)

            print("\nVerifica predizioni:")
            for i in range(3):  # Mostra 3 esempi
                year_true = np.sum(y['ordinal_years'][i])
                month_true = y['residual_months'][i][0]

                year_pred = np.sum(preds['ordinal_output'][i] > 0.5)
                month_pred = preds['month_output'][i][0]

                print(f"Esempio {i + 1}:")
                print(f"  Vero: {year_true} anni, {month_true:.1f} mesi")
                print(f"  Pred: {year_pred} anni, {month_pred:.1f} mesi")

    # Checkpoint e early stopping
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        ),
        MonitorValuesCallback()
    ]

    # Addestramento
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    return history, model