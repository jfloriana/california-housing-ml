import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def _residual_block(x, units=128):
    skip = x
    x = layers.Dense(units, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dense(units, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, skip])
    x = layers.ReLU()(x)
    return x


def build_residual_mlp(input_shape=8):
    if isinstance(input_shape, int):
        input_shape = (input_shape,)

    inputs = keras.Input(shape=input_shape)
    x = layers.Dense(128, kernel_initializer="he_normal")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = _residual_block(x, 128)
    x = _residual_block(x, 128)
    x = _residual_block(x, 128)

    x = layers.Dense(64, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Dense(32, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="residual_mlp")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae", "mse"],
    )
    return model
