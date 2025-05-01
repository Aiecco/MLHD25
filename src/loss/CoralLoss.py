import tensorflow as tf
from keras.src.saving import register_keras_serializable


@register_keras_serializable(package="Custom")
def coral_ordinal_loss(y_true_years, ord_logits):
    """
    y_true_years: [batch] valori interi in [0, max_years)
    ord_logits:   [batch, max_years] logit per ciascuna soglia
    Ritorna la somma di BCE su ciascuna soglia.
    """
    # 1) Crea una matrice [batch, max_years] di truth per ogni soglia:
    #    per ciascun sample, per soglia k,
    #    truth = 1 se y_true_years > k, 0 altrimenti.
    y_true_years = tf.cast(y_true_years, tf.int32)
    max_years = tf.shape(ord_logits)[1]
    # broadcast range [0,1,2,...,max_years-1] su batch
    thresholds = tf.range(max_years)[None, :]  # shape [1, max_years]
    y_true_thresholds = tf.cast(y_true_years[:, None] > thresholds, tf.float32)

    # 2) BCE tra logits e truth per ogni soglia
    bce = tf.keras.losses.binary_crossentropy(
        y_true_thresholds,
        tf.nn.sigmoid(ord_logits)
    )
    # 3) Media o somma su soglie e batch
    #    qui sommiamo su soglie e facciamo media sul batch
    loss = tf.reduce_mean(tf.reduce_sum(bce, axis=1))
    return loss

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
