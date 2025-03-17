import tensorflow as tf


def load_tensor(file_prefix, name):
    """Carica un tensore da un file .data e .index."""
    tensor = tf.raw_ops.RestoreV2(
        prefix=file_prefix,
        tensor_names=[name],  # Assicurati che questo corrisponda al nome del tensore salvato
        shape_and_slices=[""], #added this line
        dtypes=[tf.float32],  # Assicurati che questo corrisponda al tipo di dati del tensore
    )
    return tensor[0]
