
import tensorflow as tf
from keras.src.saving import register_keras_serializable


# Costruiamo un piccolo Sequential di preprocessing:
augmentation_layer = tf.keras.Sequential([
    # Flip orizzontale
    tf.keras.layers.RandomFlip("horizontal"),
    # Rotazione casuale ±15° → 15°/360° ≈ 0.0416
    tf.keras.layers.RandomRotation(
        factor=0.0416, fill_mode='nearest'
    ),
    # Variazione di contrasto ±10%
    tf.keras.layers.RandomContrast(0.1)
])

def augment_image(image):
    """
    Applica augmentations conservative adatte a radiografie:
      - flip orizzontale
      - rotazione casuale ±15°
      - variazione di contrasto ±10%
    """
    # I layer di preprocessing si aspettano un batch di immagini,
    # quindi espandiamo dims: [H,W,1] → [1,H,W,1], applichiamo e riduciamo
    img_batch = tf.expand_dims(image, 0)
    augmented = augmentation_layer(img_batch)
    return tf.squeeze(augmented, 0)
