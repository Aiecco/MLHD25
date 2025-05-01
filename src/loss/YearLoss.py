import tensorflow as tf
from keras.src.saving import register_keras_serializable

@register_keras_serializable(package="Custom")
def months_mae(y_true_months, y_pred_months):
    """
    y_true_months: [batch] vera età in mesi
    y_pred_months: [batch,1] età stimata in mesi dal modello
    Ritorna l'MAE medio in mesi.
    """
    # riduci a shape [batch]
    y_pred = tf.squeeze(y_pred_months, axis=-1)
    # MAE
    return tf.reduce_mean(tf.abs(y_pred - y_true_months))
