import tensorflow as tf


def months_mae(y_true_months, y_pred):
    ord_logits = y_pred[0]
    month_pred = y_pred[1]
    est_years = tf.reduce_sum(tf.cast(ord_logits > 0.5, tf.float32), axis=1)
    est_months = est_years * 12.0 + tf.squeeze(month_pred, -1)
    return tf.reduce_mean(tf.abs(est_months - y_true_months))
