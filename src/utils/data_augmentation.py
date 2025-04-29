import tensorflow as tf
from keras import layers
# Define augmentation layers outside of the function for better performance
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal", seed=42),
    layers.RandomRotation(0.15, seed=42),
    layers.RandomTranslation(0.1, 0.1, seed=42),
    layers.RandomZoom(0.1, seed=42),
])


@tf.function
def apply_augmentation(image, seed=None):
    """
    Applica data augmentation a un'immagine usando le operazioni di tf.image

    Args:
        image: Tensor immagine da aumentare
        seed: Seed per la randomizzazione (per avere trasformazioni consistenti)

    Returns:
        Tensor dell'immagine aumentata con stessa shape dell'input
    """
    # Ensure image has correct shape and dtype
    if len(tf.shape(image)) < 3:
        image = tf.expand_dims(image, axis=-1)

    # Convert to float32 if needed
    if image.dtype != tf.float32:
        image = tf.cast(image, tf.float32)

    # Add batch dimension
    img = tf.expand_dims(image, axis=0)

    # Apply augmentation
    aug_img = data_augmentation(img, training=True)

    # Remove batch dimension
    aug_img = aug_img[0]

    # Set the shape explicitly
    aug_img.set_shape(image.shape)

    return aug_img


@tf.function
def data_augmentation_pipeline(pooled_input, heatmap_input, gender_input, age, training=True):
    """
    Pipeline di data augmentation che applica le stesse trasformazioni
    a immagini pooled e heatmap correlate

    Args:
        pooled_input: Tensor of pooled image
        heatmap_input: Tensor of heatmap image
        gender_input: Tensor of gender (not augmented)
        age: Tensor of age (not augmented)
        training: Whether to apply augmentation

    Returns:
        Tuple of augmented tensors (pooled, heatmap, gender, age)
    """
    if not training:
        return pooled_input, heatmap_input, gender_input, age

    # Ensure all tensors have the correct data type
    pooled_input = tf.cast(pooled_input, tf.float32)
    heatmap_input = tf.cast(heatmap_input, tf.float32)
    gender_input = tf.cast(gender_input, tf.float32)
    age = tf.cast(age, tf.float32)

    # Apply same seed for consistent transformations
    tf.random.set_seed(42)

    # Apply augmentation to images
    pooled_aug = apply_augmentation(pooled_input)
    heatmap_aug = apply_augmentation(heatmap_input)

    # Ensure shapes are preserved
    pooled_aug.set_shape(pooled_input.shape)
    heatmap_aug.set_shape(heatmap_input.shape)

    # Gender and age don't need augmentation - return as is
    return pooled_aug, heatmap_aug, gender_input, age
