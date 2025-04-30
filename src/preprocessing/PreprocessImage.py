from src.augmentation.DataAugmentation import augment_image
import tensorflow as tf

from src.augmentation.DataAugmentation import augment_image
import tensorflow as tf


def preprocess_example(img, rad_feats, gender, true_months):
    """
    img: tensor HxWx1
    rad_feats: tensor [radiomics_dim]
    gender: scalar tensor
    true_months: scalar tensor
    """
    # augment solo su singola immagine
    img = augment_image(img)
    img = tf.clip_by_value(img, 0.0, 1.0)

    # ritorna la tripla completa come input
    return (img, rad_feats, gender), true_months