import os
import tensorflow as tf
import pandas as pd

# --- RadiographDatasetBuilder (Adjusted) ---
class RadiographDatasetBuilder:
    """
    Builds a TensorFlow Dataset for radiograph images, handling raw,
    preprocessed, and extracted image inputs along with age labels.
    """
    def __init__(self,
                 base_dir,
                 label_csv,
                 img_subfolder="prep_images",
                 prepimg_subfolder="prep_images",
                 extrimg_subfolder="extr_images",
                 img_size=(128, 128),
                 batch_size=16):
        """
        Initializes the dataset builder.

        Args:
            base_dir (str): The base directory containing image subfolders and the label CSV.
            label_csv (str): The name of the CSV file containing image IDs and bone ages.
            img_subfolder (str): Subfolder for raw images.
            prepimg_subfolder (str): Subfolder for preprocessed images.
            extrimg_subfolder (str): Subfolder for extracted images.
            img_size (tuple): Desired image size (height, width).
            batch_size (int): Batch size for the dataset.
        """
        self.base_dir = base_dir
        self.img_subfolder = os.path.join(base_dir, img_subfolder)
        self.prepimg_subfolder = os.path.join(base_dir, prepimg_subfolder)
        self.img_size = img_size
        self.batch_size = batch_size

        # Read CSV and prepare age_map
        df = pd.read_csv(os.path.join(label_csv))
        try:
            df['filename'] = df['id'].astype(str)
            df['age_months'] = df['boneage']
        except KeyError: # Handle cases where column names might be different or delimiter is ';'
            df = pd.read_csv(os.path.join(label_csv), delimiter=";")
            df['filename'] = df['id'].astype(str)
            df['age_months'] = df['boneage']

        self.age_map = dict(zip(df['filename'], df['age_months']))

    def _parse_function(self, filepath):
        """
        Parses a single image file and its corresponding preprocessed/extracted images
        and age label. This function runs in Python context via tf.py_function.

        Args:
            filepath (tf.Tensor): The TensorFlow string tensor representing the image file path.

        Returns:
            tuple: (raw_img, prep_img, extr_img, age_months) as TensorFlow tensors.
        """
        # Convert filepath tensor to Python string for file operations
        filepath_str = filepath.numpy().decode('utf-8')

        # 1) Read and normalize raw image
        raw_img = tf.io.read_file(filepath_str)
        raw_img = tf.image.decode_png(raw_img, channels=1)
        raw_img = tf.image.resize(raw_img, self.img_size) / 255.0

        # 2) Extract filename stem to find corresponding preprocessed/extracted images
        fname_tensor = tf.strings.split(filepath, os.sep)[-1]
        stem_tensor = tf.strings.split(fname_tensor, '.')[0]
        fname = stem_tensor.numpy().decode('utf-8') # Python string for dict lookup and path construction

        # Construct paths for preprocessed and extracted images
        prep_path = os.path.join(self.prepimg_subfolder, fname + ".png")

        # Read and normalize preprocessed image
        prep_img = tf.io.read_file(prep_path)
        prep_img = tf.image.decode_png(prep_img, channels=1)
        prep_img = tf.image.resize(prep_img, self.img_size) / 255.0

        # Get age from map, handling potential comma decimal separator
        try:
            age_months = float(self.age_map[fname])
        except ValueError:
            age_months = float(str(self.age_map[fname]).replace(',', '.'))

        # Return as TensorFlow tensors
        return raw_img, prep_img, age_months

    def build(self, train=True):
        """
        Builds and returns a TensorFlow Dataset.

        Args:
            train (bool): If True, shuffles the dataset.

        Returns:
            tf.data.Dataset: A TensorFlow dataset yielding
                             ((raw_img, prep_img), age_months) tuples.
        """
        pattern = os.path.join(self.img_subfolder, "*.png")
        ds = tf.data.Dataset.list_files(pattern, shuffle=train)

        def _tf_parse(fp):
            """
            TensorFlow-graph compatible wrapper for _parse_function.
            """
            img, prep_img, age_months = tf.py_function(
                func=self._parse_function,
                inp=[fp], # Pass the filepath tensor to the py_function
                Tout=[tf.float32, tf.float32, tf.float32]
            )

            # Set static shapes for the output tensors
            img.set_shape((*self.img_size, 1))
            prep_img.set_shape((*self.img_size, 1))
            age_months.set_shape(())

            # Return inputs as a tuple for the multi-input Keras model
            return prep_img, age_months

        # Map the parsing function over the dataset
        ds = ds.map(_tf_parse, num_parallel_calls=tf.data.AUTOTUNE)

        # Batch and prefetch for performance
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
