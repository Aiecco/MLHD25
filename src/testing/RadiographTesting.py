import tensorflow as tf


def test_model(model: tf.keras.Model, test_dataset: tf.data.Dataset, batch_size=16):
    """
    Valuta il modello su un dataset di test.

    Parameters:
    - model: tf.keras.Model, già compilato e caricato.
    - test_dataset: tf.data.Dataset che emette ((img, rad), (age_year, age_month, gender))
    - batch_size: batch size da usare per il test

    Returns:
    - Un dizionario con le metriche calcolate
    """
    # Preprocess del test set (stesso mapping del training)
    test_ds = (
        test_dataset
        .map(
            lambda features, labels: (
                (features[0], features[1]),  # img, radiomics
                (labels[0], labels[1], labels[2])  # age_year, age_month, gender
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Valutazione
    results = model.evaluate(test_ds, return_dict=True)
    print("\n📊 Risultati del test:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")

    return results
