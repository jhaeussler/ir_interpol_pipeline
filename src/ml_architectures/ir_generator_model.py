import tensorflow as tf


def lr_scheduler(epoch, lr):
    if epoch < 20:
        return lr
    elif lr > 1e-4:
        return lr * 0.99
    return lr


class IRGenerator(tf.Module):
    train_ds = None
    val_ds = None
    test_ds = None
    training_epochs = None
    history = None
    losses = None

    def __init__(self,
                 input_sequence_length,
                 batch_size=1,
                 feature_dims=1,
                 out_sequence_length=None,
                 output_feature_dims=None,
                 learning_rate=0.005,
                 model_struct=0,
                 max_rec_pos_index=0,
                 name=None):
        super().__init__(name=name)

        self.input_sequence_length = input_sequence_length

        self.batch_size = batch_size

        self.max_rec_pos_index = max_rec_pos_index

        self.feature_dims = feature_dims
        self.input_shape = (self.input_sequence_length, self.feature_dims)

        print(f'network input: {self.input_shape}')
        print(f'maxx pos: {self.max_rec_pos_index}')

        if out_sequence_length is None:
            self.output_sequence_length = self.input_sequence_length
        else:
            self.output_sequence_length = int(out_sequence_length)

        self.output_feature_dims = output_feature_dims

        # Define Model Structure
        if model_struct == 0:
            self.model = get_dense_model(self.input_shape,
                                               self.output_sequence_length,
                                               self.output_feature_dims,
                                               self.max_rec_pos_index)
        elif model_struct == 1:
            self.model = get_conv_model(self.input_shape,
                                        self.output_sequence_length,
                                        self.output_feature_dims,
                                        self.max_rec_pos_index)
        else:
            self.model = get_lstm_model(self.input_shape,
                                        self.output_sequence_length,
                                        self.output_feature_dims,
                                        self.max_rec_pos_index)

        self.learning_rate = learning_rate
        self.loss = tf.keras.losses.MeanSquaredError()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.model.compile(loss=self.loss, optimizer=self.optimizer, run_eagerly=True)
        self.model.summary()

        self.train_metric = tf.keras.metrics.MeanSquaredError()
        self.val_metric = tf.keras.metrics.MeanSquaredError()

        self.callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=5,
                verbose=1,
                restore_best_weights=True),

            # tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
        ]

    def train_model(self, train_ds, val_ds, epochs):
        self.train_ds = train_ds
        self.val_ds = val_ds

        self.training_epochs = epochs

        self.history = self.model.fit(self.train_ds,
                                      epochs=self.training_epochs,
                                      callbacks=self.callbacks,
                                      validation_data=self.val_ds,
                                      shuffle=True,
                                      verbose=2
                                      )

    def evaluate(self, test_ds):
        self.test_ds = test_ds
        self.losses = self.model.evaluate(self.test_ds, return_dict=True)

    def predict(self, input_sequence):
        return self.model.predict(input_sequence)


def get_dense_model(input_shape, output_sequence_length, out_feature_dims, max_pos):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape, name='input_layer'),
        tf.keras.layers.Embedding(max_pos, 64, input_length=1),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8192, name='dense_1'),
        # tf.keras.layers.Dense(8192, name='dense_2'),
        # tf.keras.layers.Dense(8192, name='dense_3'),
        # tf.keras.layers.Dense(8192, name='dense_4'),
        # tf.keras.layers.Dense(8192, name='dense_5'),
        # tf.keras.layers.Dense(8192, name='dense_6'),
        # tf.keras.layers.Dense(8192, name='dense_7'),
        # tf.keras.layers.Dense(8192, name='dense_8'),

        tf.keras.layers.Dense(output_sequence_length * out_feature_dims, name='dense_last'),
        tf.keras.layers.Reshape((output_sequence_length, out_feature_dims))
    ])
    return model


def get_conv_model(input_shape, output_sequence_length, out_feature_dims, max_pos):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape, name='input_layer'),
        tf.keras.layers.Embedding(max_pos, 64, input_length=1),
        tf.keras.layers.Reshape((64, 1)),
        tf.keras.layers.Conv1D(filters=64,
                               kernel_size=2,
                               strides=1, padding='valid', activation='relu'),
        tf.keras.layers.Conv1D(filters=64,
                               kernel_size=2,
                               dilation_rate=2,
                               strides=1, padding='valid', activation='relu'),
        tf.keras.layers.Conv1D(filters=64,
                               kernel_size=4,
                               dilation_rate=2,
                               strides=1, padding='valid', activation='relu'),
        tf.keras.layers.Conv1D(filters=64,
                               kernel_size=4,
                               dilation_rate=2,
                               strides=1, padding='valid', activation='relu'),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(8192, name='dense_1'),
        tf.keras.layers.Dense(8192, name='dense_2'),
        # tf.keras.layers.Dense(8192, name='dense_3'),
        # tf.keras.layers.Dense(8192, name='dense_4'),

        tf.keras.layers.Dense(output_sequence_length * out_feature_dims, name='dense_last'),
        tf.keras.layers.Reshape((output_sequence_length, out_feature_dims))
    ])
    return model


def get_lstm_model(input_shape, output_sequence_length, out_feature_dims, max_pos):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape, name='input_layer'),
        tf.keras.layers.Embedding(max_pos, 64, input_length=1),
        tf.keras.layers.Reshape((64, 1)),
        tf.keras.layers.LSTM(256,
                             return_sequences=False,
                             kernel_initializer=tf.keras.initializers.HeNormal(),
                             name='lstm'),
        tf.keras.layers.Dense(16384, name='dense_1'),
        tf.keras.layers.Dense(16384, name='dense_2'),
        # tf.keras.layers.Dense(16384, name='dense_3'),
        # tf.keras.layers.Dense(8192, name='dense_4'),
        # tf.keras.layers.Dense(8192, name='dense_3'),
        # tf.keras.layers.Dense(8192, name='dense_4'),

        tf.keras.layers.Dense(output_sequence_length * out_feature_dims, name='dense_last'),
        tf.keras.layers.Reshape((output_sequence_length, out_feature_dims))
    ])
    return model
