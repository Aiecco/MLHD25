from tensorflow.keras import layers, regularizers, Model, Input


class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, use_pooling=True, pool_type='avg', l2_reg=0.001):
        super(ResidualBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size, strides=strides, padding='same',
                                   kernel_regularizer=regularizers.l2(l2_reg))
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same',
                                   kernel_regularizer=regularizers.l2(l2_reg))
        self.bn2 = layers.BatchNormalization()

        self.shortcut = None
        if strides != 1:
            self.shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same',
                                          kernel_regularizer=regularizers.l2(l2_reg))

        self.relu_out = layers.ReLU()
        self.use_pooling = use_pooling
        if use_pooling:
            self.pool = layers.AveragePooling2D(2) if pool_type == 'avg' else layers.MaxPooling2D(2)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        shortcut = self.shortcut(inputs) if self.shortcut is not None else inputs
        x = layers.add([x, shortcut])
        x = self.relu_out(x)

        if self.use_pooling:
            x = self.pool(x)
        return x


def build_dual_branch_cnn(input_channels=1, img_size=(128, 128), gender_dim=1, l2_reg=0.001):
    pooled_input = Input(shape=(img_size[0], img_size[1], input_channels), name='pooled_input')
    heatmap_input = Input(shape=(img_size[0], img_size[1], input_channels), name='heatmap_input')
    gender_input = Input(shape=(gender_dim,), name='gender_input', dtype='float32')

    # Branch 1
    x1 = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu',
                       kernel_regularizer=regularizers.l2(l2_reg))(pooled_input)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.MaxPooling2D(3, strides=2, padding='same')(x1)

    for filters, strides in zip([64, 128, 256, 512], [1, 2, 2, 2]):
        x1 = ResidualBlock(filters, strides=strides, use_pooling=False, l2_reg=l2_reg)(x1)
    x1 = layers.GlobalAveragePooling2D()(x1)

    # Branch 2
    x2 = layers.Conv2D(64, 7, strides=2, padding='same', activation='relu',
                       kernel_regularizer=regularizers.l2(l2_reg))(heatmap_input)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.MaxPooling2D(3, strides=2, padding='same')(x2)

    for filters, strides in zip([64, 128, 256, 512], [1, 2, 2, 2]):
        x2 = ResidualBlock(filters, strides=strides, use_pooling=False, pool_type='max', l2_reg=l2_reg)(x2)
    x2 = layers.GlobalAveragePooling2D()(x2)

    # Merge
    merged = layers.Concatenate(axis=-1)([x1, x2, gender_input])

    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(merged)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, name='age_output')(x)

    return Model(inputs={'pooled_input': pooled_input, 'heatmap_input': heatmap_input, 'gender_input': gender_input},
                 outputs=output)


class DualBranchCNN(Model):
    def __init__(self, input_channels=1, img_size=(128, 128), gender_dim=1, l2_reg=0.001, **kwargs):
        super().__init__(**kwargs)
        self.input_channels = input_channels
        self.img_size = img_size
        self.gender_dim = gender_dim
        self.l2_reg = l2_reg
        self.model = build_dual_branch_cnn(input_channels, img_size, gender_dim, l2_reg)

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def summary(self, *args, **kwargs):
        return self.model.summary(*args, **kwargs)

    def save(self, filepath, *args, **kwargs):
        return self.model.save(filepath, *args, **kwargs)

    def load_weights(self, filepath, *args, **kwargs):
        return self.model.load_weights(filepath, *args, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_channels': self.input_channels,
            'img_size': self.img_size,
            'gender_dim': self.gender_dim,
            'l2_reg': self.l2_reg
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
