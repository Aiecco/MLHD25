import tensorflow as tf

def months_mae(y_true_months, y_pred):
    """Calcola l'errore medio assoluto in mesi per la predizione dell'età"""
    ord_logits = y_pred[0]
    month_pred = y_pred[1]
    est_years = tf.reduce_sum(tf.cast(ord_logits > 0.5, tf.float32), axis=1)
    est_months = est_years * 12.0 + tf.squeeze(month_pred, -1)
    return tf.reduce_mean(tf.abs(est_months - y_true_months))

def years_exact_acc(y_true_months, y_pred):
    """Calcola l'accuratezza (esatta) della predizione degli anni"""
    ord_logits = y_pred[0]
    est_years = tf.reduce_sum(tf.cast(ord_logits > 0.5, tf.float32), axis=1)
    true_years = tf.floor(y_true_months / 12.0)
    return tf.reduce_mean(tf.cast(tf.equal(est_years, true_years), tf.float32))

def years_within_one_acc(y_true_months, y_pred):
    """Calcola l'accuratezza della predizione degli anni entro ±1 anno"""
    ord_logits = y_pred[0]
    est_years = tf.reduce_sum(tf.cast(ord_logits > 0.5, tf.float32), axis=1)
    true_years = tf.floor(y_true_months / 12.0)
    within_one = tf.abs(est_years - true_years) <= 1
    return tf.reduce_mean(tf.cast(within_one, tf.float32))

def years_within_two_acc(y_true_months, y_pred):
    """Calcola l'accuratezza della predizione degli anni entro ±2 anni"""
    ord_logits = y_pred[0]
    est_years = tf.reduce_sum(tf.cast(ord_logits > 0.5, tf.float32), axis=1)
    true_years = tf.floor(y_true_months / 12.0)
    within_two = tf.abs(est_years - true_years) <= 2
    return tf.reduce_mean(tf.cast(within_two, tf.float32))

def months_mae(y_true_months, y_pred):
    """Calcola l'errore medio assoluto in mesi per la predizione dell'età"""
    ord_logits = y_pred[0]
    month_pred = y_pred[1]
    est_years = tf.reduce_sum(tf.cast(ord_logits > 0.5, tf.float32), axis=1)
    est_months = est_years * 12.0 + tf.squeeze(month_pred, -1)
    return tf.reduce_mean(tf.abs(est_months - y_true_months))