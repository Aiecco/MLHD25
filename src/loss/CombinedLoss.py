import tensorflow as tf

from src.loss.CoralLoss import coral_loss


def combined_loss(y_true_months, y_pred):
    # Se y_pred è un dizionario, devi usare le chiavi corrette
    ord_logits = y_pred['ord_logits']
    month_pred = y_pred['month_out']

    # Conversione età da mesi → anni interi + mesi residui
    y_true_years = tf.floor(y_true_months / 12.0)
    y_true_res = y_true_months - y_true_years * 12.0

    # Per sicurezza, rimuovi dimensioni superflue se necessario
    y_true_years = tf.squeeze(y_true_years, axis=-1) if len(y_true_years.shape) > 1 else y_true_years
    y_true_res = tf.squeeze(y_true_res, axis=-1) if len(y_true_res.shape) > 1 else y_true_res

    # Calcolo delle due loss
    loss_years = coral_loss(y_true_years, ord_logits)
    loss_month = tf.keras.losses.MSE(y_true_res, tf.squeeze(month_pred, axis=-1))

    return loss_years + 0.5 * loss_month
