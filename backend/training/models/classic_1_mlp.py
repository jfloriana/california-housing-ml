import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_mlp(input_shape=8):
    if isinstance(input_shape, int):
        input_shape = (input_shape,)

    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(64, activation="relu", kernel_initializer="he_normal")(inputs)
    x = layers.Dense(32, activation="relu", kernel_initializer="he_normal")(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="mlp")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", "mse"],
    )
    return model
