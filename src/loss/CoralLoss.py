import tensorflow as tf
from keras.src.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
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

@register_keras_serializable(package="Custom")
def combined_loss(y_true_months, y_pred):
    """
    y_true_months: [batch] età totale in mesi (es. 35 mesi -> 2 anni + 11 mesi)
    y_pred: tuple (ord_logits, month_pred)
       ord_logits: [batch, max_years] logit per soglia ordinale
       month_pred: [batch, 1] residuo mesi predetto (float)
    Restituisce ordinal_loss + mse(residual_months).
    """
    print(y_pred)
    ord_logits, month_pred = y_pred

    # 1) split età in years e mesi residui
    #    anni interi = floor(months / 12)
    #    mesi residui = months % 12
    y_true_years = tf.math.floordiv(y_true_months, 12)
    y_true_residual = tf.cast(y_true_months % 12, tf.float32)

    # 2) ordinal loss
    loss_ord = coral_ordinal_loss(y_true_years, ord_logits)

    # 3) mse sul residuo mesi
    #    month_pred ha shape [batch,1], y_true_residual [batch]
    month_pred = tf.squeeze(month_pred, axis=-1)
    loss_resid = tf.reduce_mean(tf.keras.losses.mean_squared_error(
        y_true_residual, month_pred
    ))

    # 4) somma pesata (puoi aggiustare i coefficienti se vuoi)
    return loss_ord + loss_resid
