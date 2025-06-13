# src/dataset/radiograph_dataset_builder.py
import os
import tensorflow as tf
import pandas as pd


# --- RadiographDatasetBuilder ---
class RadiographDatasetBuilder:
    """
    Builds a TensorFlow Dataset for radiograph images, handling raw and
    preprocessed image inputs along with age labels, applying pixel standardization.
    """

    def __init__(self,
                 base_dir,
                 label_csv,
                 img_subfolder="prep_images",
                 prepimg_subfolder="prep_images",
                 img_size=(128, 128),
                 batch_size=16,
                 mean_pixel_value: float = 0.0,  # Nuovo parametro per la media
                 std_pixel_value: float = 1.0):  # Nuovo parametro per la deviazione standard
        """
        Initializes the dataset builder.

        Args:
            base_dir (str): The base directory containing image subfolders.
            label_csv (str): The name of the CSV file containing image IDs and bone ages,
                             relative to base_dir.
            img_subfolder (str): Subfolder containing the main images to list files from.
            prepimg_subfolder (str): Subfolder containing the preprocessed images.
            img_size (tuple): Desired image size (height, width).
            batch_size (int): Batch size for the dataset.
            mean_pixel_value (float): Mean pixel value for standardizing image data.
            std_pixel_value (float): Standard deviation of pixel values for standardizing image data.
        """
        self.base_dir = base_dir
        self.img_subfolder = os.path.join(base_dir, img_subfolder)
        self.prepimg_subfolder = os.path.join(base_dir, prepimg_subfolder)
        self.img_size = img_size
        self.batch_size = batch_size
        self.mean_pixel_value = mean_pixel_value
        self.std_pixel_value = std_pixel_value

        # Read CSV and prepare age_map
        # Corretto: label_csv dovrebbe essere relativo a base_dir
        df = pd.read_csv(label_csv)
        try:
            df['filename'] = df['id'].astype(str)
            df['age_months'] = df['boneage']
        except KeyError:  # Handle cases where column names might be different or delimiter is ';'
            df = pd.read_csv(label_csv, delimiter=";")  # Corretto path
            df['filename'] = df['id'].astype(str)
            df['age_months'] = df['boneage'].astype(str).str.replace(',', '.').astype(float)

        self.age_map = dict(zip(df['filename'], df['age_months']))

    def _parse_function(self, filepath):
        """
        Parses a single image file and its corresponding preprocessed images
        and age label, applying pixel standardization.
        This function runs in Python context via tf.py_function.

        Args:
            filepath (tf.Tensor): The TensorFlow string tensor representing the image file path.

        Returns:
            tuple: (raw_img, prep_img, age_months) as TensorFlow tensors,
                   with images standardized.
        """

        # 2) Extract filename stem to find corresponding preprocessed images
        fname_tensor = tf.strings.split(filepath, os.sep)[-1]
        stem_tensor = tf.strings.split(fname_tensor, '.')[0]
        fname = stem_tensor.numpy().decode('utf-8')  # Python string for dict lookup and path construction

        # Construct path for preprocessed image
        prep_path = os.path.join(self.prepimg_subfolder, fname + ".png")

        # Read and resize preprocessed image
        prep_img = tf.io.read_file(prep_path)
        prep_img = tf.image.decode_png(prep_img, channels=1)  # Decode as grayscale
        prep_img = tf.image.resize(prep_img, self.img_size)

        # Convert to float32 BEFORE standardization
        prep_img = tf.cast(prep_img, tf.float32)

        # standardization
        std_val_safe = self.std_pixel_value if self.std_pixel_value > 1e-7 else 1.0

        prep_img = (prep_img - self.mean_pixel_value) / std_val_safe

        # Get age from map, handling potential comma decimal separator
        try:
            age_months = float(self.age_map[fname])
        except ValueError:
            age_months = float(str(self.age_map[fname]).replace(',', '.'))

        # Return all three items. The build method's _tf_parse will then select what to pass.
        return prep_img, age_months

    def build(self, train=True):
        """
        Builds and returns a TensorFlow Dataset.

        Args:
            train (bool): If True, shuffles the dataset.

        Returns:
            tf.data.Dataset: A TensorFlow dataset yielding
                             (prep_img, age_months) tuples, as per the model's current input.
        """
        pattern = os.path.join(self.img_subfolder, "*.png")
        ds = tf.data.Dataset.list_files(pattern, shuffle=train)

        def _tf_parse(fp):
            """
            TensorFlow-graph compatible wrapper for _parse_function.
            """
            # _parse_function restituisce (raw_img, prep_img, age_months)
            img_prep, age_months = tf.py_function(
                func=self._parse_function,
                inp=[fp],  # Pass the filepath tensor to the py_function
                Tout=[tf.float32, tf.float32]  # Tipi di output di _parse_function
            )

            # Set static shapes for the output tensors
            img_prep.set_shape((*self.img_size, 1))
            age_months.set_shape(())

            return img_prep, age_months  # Restituisce solo l'immagine preprocessata e l'età

        # Map the parsing function over the dataset
        ds = ds.map(_tf_parse, num_parallel_calls=tf.data.AUTOTUNE)

        # Batch and prefetch for performance
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds
