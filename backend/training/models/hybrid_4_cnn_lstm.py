import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_cnn_lstm(input_shape=8):
    if isinstance(input_shape, int):
        input_shape = (input_shape,)

    inputs = keras.Input(shape=input_shape)
    x = layers.Reshape((input_shape[0], 1))(inputs)
    x = layers.Conv1D(64, kernel_size=3, padding="same", kernel_initializer="he_normal")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.LSTM(32, kernel_initializer="he_normal")(x)
    x = layers.Dense(16, activation="relu", kernel_initializer="he_normal")(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="cnn_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", "mse"],
    )
    return model
