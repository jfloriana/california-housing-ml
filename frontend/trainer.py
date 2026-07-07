import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

MODEL_NAMES = [
    "MLP Baseline",
    "Deep MLP",
    "Residual MLP",
    "CNN-LSTM",
    "Autoencoder-MLP",
]

def _build_model(name, input_dim):

    if name == "MLP Baseline":
        i = keras.Input(shape=(input_dim,))
        x = layers.Dense(64, activation="relu")(i)
        x = layers.Dense(32, activation="relu")(x)
        o = layers.Dense(1)(x)
        model = keras.Model(i, o, name="mlp")

    elif name == "Deep MLP":
        i = keras.Input(shape=(input_dim,))
        x = layers.Dense(128, activation="relu")(i)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dense(16, activation="relu")(x)
        o = layers.Dense(1)(x)
        model = keras.Model(i, o, name="deep_mlp")

    elif name == "Residual MLP":
        i = keras.Input(shape=(input_dim,))
        x = layers.Dense(128, activation="relu")(i)
        skip = x
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Add()([x, skip])
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(32, activation="relu")(x)
        o = layers.Dense(1)(x)
        model = keras.Model(i, o, name="residual_mlp")

    elif name == "CNN-LSTM":
        i = keras.Input(shape=(input_dim,))
        x = layers.Reshape((input_dim, 1))(i)
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(2)(x)
        x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Reshape((x.shape[1], x.shape[2]))(x)
        x = layers.LSTM(32, return_sequences=False)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(16, activation="relu")(x)
        o = layers.Dense(1)(x)
        model = keras.Model(i, o, name="cnn_lstm")

    elif name == "Autoencoder-MLP":
        i = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(8, activation="relu")(i)
        encoded = layers.Dense(4, activation="relu")(encoded)
        decoded = layers.Dense(8, activation="relu")(encoded)
        decoded = layers.Dense(input_dim, activation="linear")(decoded)
        regression = layers.Dense(16, activation="relu")(encoded)
        regression = layers.Dense(1)(regression)
        model = keras.Model(i, [regression, decoded], name="autoencoder_mlp")

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="mse",
        metrics=["mae"],
    )
    return model

def train_all(progress_callback=None, status_callback=None):
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.columns = ["MedInc","HouseAge","AveRooms","AveBedrms",
                   "Population","AveOccup","Latitude","Longitude","MedHouseVal"]
    X = df.drop("MedHouseVal", axis=1).values
    y = df["MedHouseVal"].values

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/0.85, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    results = []
    total = len(MODEL_NAMES)

    for idx, name in enumerate(MODEL_NAMES):
        if status_callback:
            status_callback(f"Entrenando {name} ({idx+1}/{total})...")
        if progress_callback:
            progress_callback(idx / total)

        model = _build_model(name, X_train.shape[1])
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=0
        )
        start = time.time()

        if name == "Autoencoder-MLP":
            model.fit(
                X_train_s, [y_train, X_train_s],
                validation_data=(X_val_s, [y_val, X_val_s]),
                epochs=100, batch_size=32, callbacks=[es], verbose=0,
            )
            y_pred = model.predict(X_test_s, verbose=0)[0].ravel()
        else:
            model.fit(
                X_train_s, y_train,
                validation_data=(X_val_s, y_val),
                epochs=100, batch_size=32, callbacks=[es], verbose=0,
            )
            y_pred = model.predict(X_test_s, verbose=0).ravel()

        elapsed = time.time() - start
        mse = float(mean_squared_error(y_test, y_pred))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        results.append({
            "model_name": name,
            "model_type": "Hybrid" if name in ("CNN-LSTM", "Autoencoder-MLP") else "Classic",
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4),
            "training_time_sec": round(elapsed, 2),
            "num_params": int(model.count_params()),
        })

        if status_callback:
            status_callback(f"{name}: R²={r2:.4f}, RMSE={rmse:.4f} ({elapsed:.1f}s)")
        if progress_callback:
            progress_callback((idx + 1) / total)

    results.sort(key=lambda r: r["r2"], reverse=True)
    return results, scaler
