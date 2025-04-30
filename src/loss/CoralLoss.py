import tensorflow as tf

def coral_loss(y_true_years, logits):
    """Loss CORAL per la classificazione ordinale degli anni"""
    if len(y_true_years.shape) == 2 and y_true_years.shape[-1] == 1:
        y_true_years = tf.squeeze(y_true_years, axis=-1)

    levels = tf.cast(tf.range(1, logits.shape[-1] + 1)[tf.newaxis, :], tf.float32)
    y_true_rep = tf.tile(tf.expand_dims(y_true_years, -1), [1, logits.shape[-1]])
    target = tf.cast(y_true_rep >= levels, tf.float32)  # target binario

    # Applichiamo sigmoid prima di loss se non è già nel modello
    probs = tf.sigmoid(logits)
    bce = tf.keras.losses.binary_crossentropy(target, probs)

    return tf.reduce_mean(bce)
