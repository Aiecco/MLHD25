import tensorflow as tf
from tensorflow.keras import layers

class DualBranchCNN(tf.keras.Model):
    def __init__(self, input_channels=1, img_size=(128, 128), gender_dim=0, **kwargs):
        """
        input_channels: Numero di canali dell'immagine (es. 1 per radiografie in scala di grigi)
        img_size: Dimensioni dell'immagine in input (altezza, larghezza)
        gender_dim: Dimensione del vettore che rappresenta il genere (se lo integri come feature)
        """
        super(DualBranchCNN, self).__init__(**kwargs)  # Aggiungi kwargs al super().__init__()

        # --- Branch 1: Immagini Pooled ---
        self.branch1 = tf.keras.Sequential([
            layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu',
                          input_shape=(img_size[0], img_size[1], input_channels)),
            layers.BatchNormalization(),
            layers.AveragePooling2D(2),
            layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.AveragePooling2D(2),
            layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.AveragePooling2D(2),
        ])

        # --- Branch 2: Heatmaps ---
        self.branch2 = tf.keras.Sequential([
            layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu',
                          input_shape=(img_size[0], img_size[1], input_channels)),
            layers.BatchNormalization(),
            layers.MaxPool2D(2),
            layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(2),
            layers.Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPool2D(2),
        ])

        # Calcolo dimensione feature dopo convoluzioni e pooling
        pooled_H, pooled_W = img_size[0] // 4, img_size[1] // 4
        branch_feat_dim = 32 * pooled_H * pooled_W

        # Layer fully-connected per la fusione delle due branche (e il genere, se fornito)
        total_feat_dim = branch_feat_dim * 2 + gender_dim

        self.fc = tf.keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(total_feat_dim,)),
            layers.Dropout(0.4),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1)
        ])
        self.input_channels = input_channels
        self.img_size = img_size
        self.gender_dim = gender_dim

    def call(self, pooled_input, heatmap_input, gender_input=None):
        # Elaborazione Branch 1
        x1 = self.branch1(pooled_input)
        x1 = tf.reshape(x1, [tf.shape(x1)[0], -1])

        # Elaborazione Branch 2
        x2 = self.branch2(heatmap_input)
        x2 = tf.reshape(x2, [tf.shape(x2)[0], -1])

        # Concatenazione delle feature
        if gender_input is not None:
            gender_input = tf.reshape(gender_input, [-1, 2])  # Assumendo che sia one-hot
            gender_input = tf.tile(gender_input, [2, 1])  # Se batch size è la metà, raddoppia

        x = tf.concat([x1, x2, gender_input], axis=1)

        # Passaggio attraverso la rete fully-connected
        age_output = self.fc(x)

        return age_output

    def get_config(self):
        config = super(DualBranchCNN, self).get_config()
        config.update({
            'input_channels': self.input_channels,
            'img_size': self.img_size,
            'gender_dim': self.gender_dim,
        })
        return config