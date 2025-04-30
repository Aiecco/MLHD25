import tensorflow as tf

# Funzione per calcolare mesi totali (uguale a quella già definita)
def calculate_total_months(ord_logits, month_pred):
    est_years = tf.reduce_sum(tf.cast(ord_logits > 0.5, tf.float32), axis=1)
    est_months = est_years * 12.0 + tf.squeeze(month_pred, -1)
    return est_months