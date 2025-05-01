import tensorflow as tf

from src.plot.AGECallback import AgeMetricsCallback
from src.plot.EpochPlotCallback import EpochPlotCallback


def train_model(model, dataset, epochs=30, batch_size=16, validation_data=None):
    """
    model: tf.keras.Model già compilato con 2 input heads e 2 loss
    dataset: tf.data.Dataset che emette ((img, rad), (age_year, age_month, gender))
    validation_data: stessa struttura di `dataset`
    """
    # 1) Preprocess & batch per il train set
    train_ds = (
        dataset
        .map(
            lambda features, labels: (
                (features[0], features[1]),  # img, radiomics, gender_input
                (labels[0], labels[1])                     # age_year, age_month, gender
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
                    (labels[0], labels[1]),
                ),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
    else:
        val_ds = None

    # 3) Callback per il plot
    plot_cb = EpochPlotCallback()
    # Crea il callback
    age_metrics_callback = AgeMetricsCallback(validation_data=val_ds, frequency=5)

    # 4) Fit col validation_data
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[age_metrics_callback, plot_cb]
    )

    return model, age_metrics_callback