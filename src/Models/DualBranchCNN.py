import tensorflow as tf
from tensorflow.keras import layers

class DualBranchCNN(tf.keras.Model):
    def __init__(self, input_channels=1, img_size=(128, 128), gender_dim=0):
        """
        input_channels: Numero di canali dell'immagine (es. 1 per radiografie in scala di grigi)
        img_size: Dimensioni dell'immagine in input (altezza, larghezza)
        gender_dim: Dimensione del vettore che rappresenta il genere (se lo integri come feature)
        """
        super(DualBranchCNN, self).__init__()

        # --- Branch 1: Immagini Pooled ---
        self.branch1 = tf.keras.Sequential([
            layers.Conv2D(16, kernel_size=3, strides=1, padding='same', input_shape=(img_size[0], img_size[1], input_channels)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(2),
            layers.Conv2D(32, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(2)
        ])

        # --- Branch 2: Heatmaps ---
        self.branch2 = tf.keras.Sequential([
            layers.Conv2D(16, kernel_size=3, strides=1, padding='same', input_shape=(img_size[0], img_size[1], input_channels)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(2),
            layers.Conv2D(32, kernel_size=3, strides=1, padding='same'),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPool2D(2)
        ])

        # Calcolo dimensione feature dopo convoluzioni e pooling
        pooled_H, pooled_W = img_size[0] // 4, img_size[1] // 4
        branch_feat_dim = 32 * pooled_H * pooled_W

        # Layer fully-connected per la fusione delle due branche (e il genere, se fornito)
        total_feat_dim = branch_feat_dim * 2 + gender_dim

        self.fc = tf.keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(total_feat_dim,)),
            layers.Dropout(0.5),
            layers.Dense(1)
        ])

    def call(self, pooled_input, heatmap_input, gender_input=None):
        # Elaborazione Branch 1
        x1 = self.branch1(pooled_input)
        x1 = tf.reshape(x1, [tf.shape(x1)[0], -1])

        # Elaborazione Branch 2
        x2 = self.branch2(heatmap_input)
        x2 = tf.reshape(x2, [tf.shape(x2)[0], -1])

        # Concatenazione delle feature
        if gender_input is not None:
            gender_input = tf.expand_dims(gender_input, axis=1)
            x = tf.concat([x1, x2, gender_input], axis=1)
        else:
            x = tf.concat([x1, x2], axis=1)

        # Passaggio attraverso la rete fully-connected
        age_output = self.fc(x)
        return age_output