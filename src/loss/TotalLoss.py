from src.loss.CombinedLoss import combined_loss
import tensorflow as tf


def total_loss(y_true_months, y_pred, y_true_gender, y_pred_gender):
    # age-part
    loss_age = combined_loss(y_true_months, y_pred)
    # gender-part (binary crossentropy)
    loss_gender = tf.keras.losses.binary_crossentropy(y_true_gender, y_pred_gender)
    return loss_age + loss_gender  # GRL si occuperà di invertire il gradient per backbone
