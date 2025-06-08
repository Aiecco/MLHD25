import tensorflow as tf
from keras import layers, regularizers, models

from src.Models.AttentionLayer import SpatialAttention

# --- Age Prediction Model with Attention ---
class AgePredictionModel:
    def __init__(self, img_size=(128, 128)):
        self.img_size = img_size
        self.model = self._build_model()

    def _create_cnn_branch(self, input_tensor, name_prefix):
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                          kernel_regularizer=regularizers.l2(1e-4), name=f'{name_prefix}_conv1a')(input_tensor)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1a')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu1a')(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                          kernel_regularizer=regularizers.l2(1e-4), name=f'{name_prefix}_conv1b')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1b')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu1b')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=f'{name_prefix}_pool1')(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                          kernel_regularizer=regularizers.l2(1e-4), name=f'{name_prefix}_conv2a')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2a')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu2a')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                          kernel_regularizer=regularizers.l2(1e-4), name=f'{name_prefix}_conv2b')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2b')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu2b')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=f'{name_prefix}_pool2')(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                          kernel_regularizer=regularizers.l2(1e-4), name=f'{name_prefix}_conv3a')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn3a')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu3a')(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                          kernel_regularizer=regularizers.l2(1e-4), name=f'{name_prefix}_conv3b')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn3b')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu3b')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=f'{name_prefix}_pool3')(x)

        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                          kernel_regularizer=regularizers.l2(1e-4), name=f'{name_prefix}_conv4a')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn4a')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu4a')(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                          kernel_regularizer=regularizers.l2(1e-4), name=f'{name_prefix}_conv4b')(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn4b')(x)
        x = layers.Activation('relu', name=f'{name_prefix}_relu4b')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name=f'{name_prefix}_pool4')(x)

        x = layers.Conv2D(256, (3, 3), padding='same', name='conv5a')(x)
        x = layers.BatchNormalization(name='bn5a')(x)
        x = layers.Activation('relu', name='relu5a')(x)
        x = layers.Conv2D(256, (3, 3), padding='same', name='conv5b')(x)
        x = layers.BatchNormalization(name='bn5b')(x)
        x = layers.Activation('relu', name='relu5b')(x)
        x = layers.MaxPooling2D((2, 2), name='pool5')(x)

        return x

    def _build_model(self):
        prep_input = layers.Input(shape=(*self.img_size, 1), name='prep_input')

        prep_features = self._create_cnn_branch(prep_input, 'prep')

        attended_prep_features = SpatialAttention(name='attention_prep')(prep_features)

        x = layers.Flatten(name='flatten_features')(attended_prep_features)

        # Primo strato Dense: aumentato il numero di neuroni
        x = layers.Dense(512, activation='relu', name='dense1',  # Aumentato a 512
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization(name='bn_dense1')(x)
        x = layers.Dropout(0.4, name='dropout1')(x)  # Leggermente ridotto il dropout

        # Secondo strato Dense: aumentato il numero di neuroni
        x = layers.Dense(256, activation='relu', name='dense2',  # Aumentato a 256
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization(name='bn_dense2')(x)
        x = layers.Dropout(0.4, name='dropout2')(x)  # Leggermente ridotto il dropout

        # Nuovo terzo strato Dense: aggiunta di complessità
        x = layers.Dense(128, activation='relu', name='dense3',  # Nuovo strato
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization(name='bn_dense3')(x)
        x = layers.Dropout(0.3, name='dropout3')(x)  # Dropout ancora più basso qui

        # Output layer finale per la regressione
        output = layers.Dense(1, activation='linear', name='age_output')(x)

        model = models.Model(inputs=prep_input, outputs=output, name='AgePredictionModel')
        return model

    def compile_model(self, learning_rate=0.0005):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])