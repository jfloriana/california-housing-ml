import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder_mlp(input_shape=8):
    if isinstance(input_shape, int):
        input_shape = (input_shape,)

    inputs = keras.Input(shape=input_shape)

    # Encoder
    x = layers.Dense(6, activation="relu", kernel_initializer="he_normal")(inputs)
    encoded = layers.Dense(4, activation="relu", kernel_initializer="he_normal", name="encoded")(x)

    # Decoder (for autoencoder reconstruction loss)
    x = layers.Dense(6, activation="relu", kernel_initializer="he_normal")(encoded)
    decoded = layers.Dense(input_shape[0], activation="linear", name="decoded")(x)

    # Regressor branch
    x = layers.Dense(32, activation="relu", kernel_initializer="he_normal")(encoded)
    x = layers.Dense(16, activation="relu", kernel_initializer="he_normal")(x)
    regression_output = layers.Dense(1, activation="linear", name="regression_output")(x)

    model = keras.Model(
        inputs=inputs,
        outputs=[regression_output, decoded],
        name="autoencoder_mlp",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={"regression_output": "mse", "decoded": "mse"},
        loss_weights={"regression_output": 1.0, "decoded": 0.1},
        metrics={"regression_output": ["mae", "mse"]},
    )
    return model
