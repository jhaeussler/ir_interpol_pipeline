import tensorflow as tf


def lr_scheduler(epoch, lr):
    if epoch < 20:
        return lr
    elif lr > 1e-4:
        return lr * 0.99
    return lr


# Reset States for LSTM
class CustomCallbacks(tf.keras.callbacks.Callback):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_epoch_end(self, *args):
        self.model.reset_states()

    def on_batch_end(self, *args, **kwargs):
        self.model.reset_states()


class ReverbGenerator(tf.Module):
    train_ds = None
    val_ds = None
    test_ds = None
    training_epochs = None
    history = None
    losses = None

    def __init__(self,
                 input_sequence_length,
                 feature_dims=1,
                 out_sequence_length=None,
                 learning_rate=0.005,
                 model_struct=0,
                 name=None):
        super().__init__(name=name)

        self.input_sequence_length = input_sequence_length

        self.feature_dims = feature_dims
        self.input_shape = (self.input_sequence_length, self.feature_dims)
        print(f'input_shape: {self.input_shape}')

        if out_sequence_length is None:
            self.output_sequence_length = self.input_sequence_length
        else:
            self.output_sequence_length = int(out_sequence_length)

        # Define Model Structure
        if model_struct == 0:
            self.model = get_dense_model(self.input_shape, self.output_sequence_length)
        elif model_struct == 1:
            self.model = get_conv_model_audio(self.input_shape, self.output_sequence_length)
        else:
            self.model = get_lstm_model_audio(self.input_shape, self.output_sequence_length)

        self.learning_rate = learning_rate
        self.loss = tf.keras.losses.MeanSquaredError()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

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

            CustomCallbacks(model=self.model)
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


def get_lstm_model_audio(input_shape, output_sequence_length):
    feature_dims = input_shape[1]
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape, name='input_layer'),
        tf.keras.layers.Reshape((32, int(input_shape[0] / 32))),
        tf.keras.layers.LSTM(1024,
                             return_sequences=False,
                             kernel_initializer=tf.keras.initializers.HeNormal(),
                             name='lstm'),
        tf.keras.layers.Dense(8192, name='dense_first'),
        tf.keras.layers.Dense(8192, name='dense_second'),
        # tf.keras.layers.Dense(8192, name='dense_lastt'),
        # tf.keras.layers.Dense(8192, name='dense_lasttt'),
        tf.keras.layers.Dense(output_sequence_length * feature_dims, name='dense_last'),
        tf.keras.layers.Reshape((output_sequence_length, feature_dims))
    ])
    return model


def get_dense_model(input_shape, output_sequence_length):
    feature_dims = input_shape[1]

    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape, name='input_layer'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8192, name='dense1'),
        # tf.keras.layers.Dense(8192, name='dense2'),
        # tf.keras.layers.Dense(4096, name='dense3'),
        # tf.keras.layers.Dense(4096, name='dense4'),
        # tf.keras.layers.Dense(8192, name='dense5'),
        # tf.keras.layers.Dense(8192, name='dense6'),
        # tf.keras.layers.Dense(8192, name='dense7'),
        # tf.keras.layers.Dense(8192, name='dense8'),
        tf.keras.layers.Dense(output_sequence_length * feature_dims, name='dense_last'),
        tf.keras.layers.Reshape((output_sequence_length, feature_dims))
    ])
    return model


def get_conv_model_audio(input_shape, output_sequence_length):
    feature_dims = input_shape[1]
    sequence_length = input_shape[0]

    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape, name='input_layer'),
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=4,
                               strides=1, padding='valid', activation='relu'),
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=2,
                               dilation_rate=2,
                               strides=1, padding='valid', activation='relu'),
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=2,
                               dilation_rate=4,
                               strides=1, padding='valid', activation='relu'),
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=2,
                               dilation_rate=8,
                               strides=1, padding='valid', activation='relu'),
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=2,
                               dilation_rate=16,
                               strides=1, padding='valid', activation='relu'),
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=2,
                               dilation_rate=32,
                               strides=1, padding='valid', activation='relu'),
        tf.keras.layers.Conv1D(filters=64,
                               kernel_size=2,
                               dilation_rate=64,
                               strides=1, padding='valid', activation='relu'),
        tf.keras.layers.Conv1D(filters=64,
                               kernel_size=2,
                               dilation_rate=128,
                               strides=1, padding='valid', activation='relu'),
        tf.keras.layers.Conv1D(filters=64,
                               kernel_size=2,
                               dilation_rate=256,
                               strides=1, padding='valid', activation='relu'),
        tf.keras.layers.Conv1D(filters=64,
                               kernel_size=2,
                               dilation_rate=512,
                               strides=1, padding='valid', activation='relu'),
        tf.keras.layers.Conv1D(filters=128,
                               kernel_size=2,
                               strides=2, padding='valid', activation='relu'),

        # tf.keras.layers.MaxPool1D(pool_size=2, strides=None, padding='valid', data_format=None),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(8192, name='dense_first'),
        tf.keras.layers.Dense(8192, name='dense_first2'),
        tf.keras.layers.Dense(8192, name='dense_first22'),
        tf.keras.layers.Dense(8192, name='dense_first222'),
        tf.keras.layers.Dense(output_sequence_length * feature_dims, name='dense_last'),
        tf.keras.layers.Reshape((output_sequence_length, feature_dims))
    ])
    return model


def get_lstm_model(input_shape, output_sequence_length):
    feature_dims = input_shape[1]
    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape, name='input_layer'),
        tf.keras.layers.LSTM(512,
                             return_sequences=False,
                             kernel_initializer=tf.keras.initializers.HeNormal(),
                             name='lstm'),
        tf.keras.layers.Dense(8192, name='dense_first'),
        tf.keras.layers.Dense(8192, name='dense_second'),
        # tf.keras.layers.Dense(4096, name='dense_third'),
        # tf.keras.layers.Dense(4096, name='dense_fourth'),
        tf.keras.layers.Dense(output_sequence_length * feature_dims, name='dense_last'),
        tf.keras.layers.Reshape((output_sequence_length, feature_dims))
    ])
    return model


def get_conv_model(input_shape, output_sequence_length):
    feature_dims = input_shape[1]
    sequence_length = input_shape[0]

    model = tf.keras.Sequential([
        tf.keras.Input(shape=input_shape, name='input_layer'),
        tf.keras.layers.Reshape((sequence_length, int(feature_dims / 2), 2)),
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=2,
                               strides=(1, 1), padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(filters=64,
                               kernel_size=2,
                               dilation_rate=2,
                               strides=(1, 1), padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(filters=128,
                               kernel_size=2,
                               dilation_rate=2,
                               strides=(1, 1), padding='valid', activation='relu'),
        tf.keras.layers.Conv2D(filters=128,
                               kernel_size=2,
                               dilation_rate=2,
                               strides=(1, 1), padding='valid', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16384, name='dense_first'),
        tf.keras.layers.Dense(2 * 16384, name='dense_second'),
        # tf.keras.layers.Dense(8192, name='dense_third'),
        # tf.keras.layers.Dense(8192, name='dense_fourth'),
        # tf.keras.layers.Dense(8192, name='dense_fifth'),
        # tf.keras.layers.Dense(8192, name='dense_sixth'),

        tf.keras.layers.Dense(output_sequence_length * feature_dims, name='dense_last'),
        tf.keras.layers.Reshape((output_sequence_length, feature_dims))
    ])
    return model
