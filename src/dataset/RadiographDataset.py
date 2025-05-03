import os
import tensorflow as tf
import pandas as pd

from src.radiomics.RadiomicsFeature import load_features  # carica un .npy → numpy array

class RadiographDatasetBuilder:
    def __init__(self,
                 base_dir,
                 label_csv,
                 img_subfolder="",      # es. "train_samples"
                 img_size=(128, 128),
                 batch_size=16):
        self.base_dir = base_dir
        self.img_subfolder = os.path.join(base_dir, img_subfolder)
        self.img_size = img_size
        self.batch_size = batch_size

        # Leggi CSV
        df = pd.read_csv(os.path.join(base_dir, label_csv))
        try:
            df['filename'] = df['id'].astype(str)
            df['age_months'] = df['boneage']
            df['gender'] = df['sex'].map({'F': 0.0, 'M': 1.0})
        except:
            df = pd.read_csv(os.path.join(base_dir, label_csv), delimiter=";")
            df['filename'] = df['id'].astype(str)
            df['age_months'] = df['boneage']
            df['gender'] = df['sex'].map({'F': 0.0, 'M': 1.0})


        self.age_map = dict(zip(df['filename'], df['age_months']))
        self.gender_map = dict(zip(df['filename'], df['gender']))

    def _parse_function(self, filepath):
        # filepath: string path to .png
        # 1) Leggi e normalizza immagine
        img = tf.io.read_file(filepath)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, self.img_size) / 255.0

        # 2) Filename, age, gender
        fname_tensor = tf.strings.split(filepath, os.sep)[-1]
        # Rimuovi l'estensione .png
        stem_tensor = tf.strings.split(fname_tensor, '.')[0]
        fname = stem_tensor.numpy().decode()  # Esegui .numpy() e .decode() solo qui

        try:
            age_months = float(self.age_map[fname])
        except ValueError:
            age_months = float(self.age_map[fname].replace(',', '.'))

        age_years = age_months // 12
        gender = float(self.gender_map[fname])

        # 3) Carica radiomics precomputate
        rad_path = tf.strings.join([self.base_dir, "/radiomics/", stem_tensor, ".npy"])
        rad_feats = tf.numpy_function(func=load_features, inp=[rad_path], Tout=tf.float32)
        rad_feats.set_shape((38,))

        # Ritorna 4 valori piatti: img, rad_feats, gender, age_year, age_month
        return img, rad_feats, gender, age_months, age_years

    def build(self, shuffle=True):
        pattern = os.path.join(self.img_subfolder, "*.png")
        ds = tf.data.Dataset.list_files(pattern, shuffle=shuffle)

        def _tf_parse(fp):
            # tf.py_function deve restituire un elenco piatto
            img, rad, gender, age_months, age_years = tf.py_function(
                func=self._parse_function,
                inp=[fp],
                Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32]
            )
            # Imposta le shape statiche su ciascun tensor
            img.set_shape((*self.img_size, 1))
            rad.set_shape((38,))
            gender.set_shape(())
            age_months.set_shape(())
            age_years.set_shape(())
            # Ricompone la struttura
            return (img, rad), (age_years, age_months, gender)

        ds = ds.map(_tf_parse, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(100)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds
