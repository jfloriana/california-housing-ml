import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

_LANG: Dict[str, Dict[str, str]] = {
    "es": {
        "tuning_title": "Optimización de Hiperparámetros - {name}",
        "trial": "Prueba {n}/{total}",
        "objective": "MSE de Validación",
        "best_params": "Mejores Hiperparámetros",
        "best_mse": "Mejor MSE",
        "best_r2": "Mejor R²",
        "top_3": "Top 3 Configuraciones",
        "rank": "Ranking",
        "learning_rate": "Tasa de Aprendizaje",
        "batch_size": "Tamaño de Lote",
        "units": "Unidades por Capa",
        "dropout_rate": "Tasa de Dropout",
        "num_layers": "Número de Capas",
        "mse": "MSE",
        "r2": "R²",
        "history_title": "Historial de Optimización",
        "importance_title": "Importancia de Hiperparámetros",
        "importance": "Importancia",
        "optuna_missing": "Optuna no está instalado. Instálelo con: pip install optuna",
        "summary_intro": "=== Resultados de Optimización de Hiperparámetros ===",
        "summary_best": "Mejor MSE: {mse:.6f}, R²: {r2:.4f}",
        "summary_trials": "Pruebas realizadas: {n}",
        "plots_saved": "Gráficos guardados en: {dir}",
        "no_trials": "No se pudieron completar pruebas de optimización.",
    },
    "en": {
        "tuning_title": "Hyperparameter Tuning - {name}",
        "trial": "Trial {n}/{total}",
        "objective": "Validation MSE",
        "best_params": "Best Hyperparameters",
        "best_mse": "Best MSE",
        "best_r2": "Best R²",
        "top_3": "Top 3 Configurations",
        "rank": "Rank",
        "learning_rate": "Learning Rate",
        "batch_size": "Batch Size",
        "units": "Units per Layer",
        "dropout_rate": "Dropout Rate",
        "num_layers": "Number of Layers",
        "mse": "MSE",
        "r2": "R²",
        "history_title": "Optimization History",
        "importance_title": "Hyperparameter Importance",
        "importance": "Importance",
        "optuna_missing": "Optuna is not installed. Install it with: pip install optuna",
        "summary_intro": "=== Hyperparameter Tuning Results ===",
        "summary_best": "Best MSE: {mse:.6f}, R²: {r2:.4f}",
        "summary_trials": "Trials completed: {n}",
        "plots_saved": "Plots saved to: {dir}",
        "no_trials": "No optimization trials could be completed.",
    },
}


def _tr(lang: str, key: str, **fmt: Any) -> str:
    val = _LANG.get(lang, _LANG["es"]).get(key, key)
    return val.format(**fmt) if fmt else val


def _save_plot(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _build_tuning_model(
    model_name: str,
    input_shape: int,
    learning_rate: float,
    units: int,
    dropout_rate: float,
    num_layers: int,
) -> keras.Model:
    model_name = model_name.lower().replace("-", "_").replace(" ", "_")
    inputs = keras.Input(shape=(input_shape,))

    if "cnn_lstm" in model_name:
        x = layers.Reshape((input_shape, 1))(inputs)
        x = layers.Conv1D(units, kernel_size=3, padding="same", kernel_initializer="he_normal")(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        x = layers.LSTM(max(16, units // 2), kernel_initializer="he_normal")(x)
        x = layers.Dense(max(8, units // 4), activation="relu", kernel_initializer="he_normal")(x)
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(1, activation="linear")(x)
    elif "autoencoder" in model_name:
        encoding_dim = max(2, units // 4)
        x = layers.Dense(units, activation="relu", kernel_initializer="he_normal")(inputs)
        for i in range(num_layers - 1):
            x = layers.Dense(units // (2 if i == 0 else 1), activation="relu", kernel_initializer="he_normal")(x)
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate)(x)
        encoded = layers.Dense(encoding_dim, activation="relu", kernel_initializer="he_normal", name="encoded")(x)
        decoded = layers.Dense(input_shape, activation="linear", name="decoded")(encoded)
        x = layers.Dense(max(8, units // 2), activation="relu", kernel_initializer="he_normal")(encoded)
        for i in range(num_layers - 1):
            x = layers.Dense(max(4, units // (4 * (i + 1))), activation="relu", kernel_initializer="he_normal")(x)
        regression_output = layers.Dense(1, activation="linear", name="regression_output")(x)
        model = keras.Model(inputs=inputs, outputs=[regression_output, decoded], name="autoencoder_tuning")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss={"regression_output": "mse", "decoded": "mse"},
            loss_weights={"regression_output": 1.0, "decoded": 0.1},
            metrics={"regression_output": ["mae"]},
        )
        return model
    else:
        x = inputs
        for i in range(num_layers):
            if "deep_mlp" in model_name or "residual" in model_name:
                x = layers.Dense(units, kernel_initializer="he_normal")(x)
                x = layers.BatchNormalization()(x)
                x = layers.ReLU()(x)
            else:
                x = layers.Dense(units, activation="relu", kernel_initializer="he_normal")(x)
            if dropout_rate > 0:
                x = layers.Dropout(dropout_rate)(x)

        if "residual" in model_name:
            shortcut = layers.Dense(units, kernel_initializer="he_normal")(inputs)
            shortcut = layers.BatchNormalization()(shortcut)
            shortcut = layers.ReLU()(shortcut)
            x = layers.Add()([x, shortcut])
            x = layers.ReLU()(x)

        outputs = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    return model


def _objective(
    trial: optuna.trial.Trial,
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_shape: int,
    is_ae: bool,
) -> float:
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    bs = trial.suggest_categorical("batch_size", [16, 32, 64])
    units = trial.suggest_categorical("units", [32, 64, 128])
    dropout = trial.suggest_float("dropout_rate", 0.1, 0.5)
    num_layers = trial.suggest_int("num_layers", 2, 5)

    model = _build_tuning_model(model_name, input_shape, lr, units, dropout, num_layers)

    es = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True, verbose=0
    )

    try:
        if is_ae:
            model.fit(
                X_train, [y_train, X_train],
                validation_data=(X_val, [y_val, X_val]),
                epochs=30, batch_size=bs,
                callbacks=[es], verbose=0,
            )
            y_pred = model.predict(X_val, verbose=0)[0].ravel()
        else:
            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30, batch_size=bs,
                callbacks=[es], verbose=0,
            )
            y_pred = model.predict(X_val, verbose=0).ravel()

        val_mse = float(mean_squared_error(y_val, y_pred))
    except Exception:
        val_mse = float("inf")
    finally:
        keras.backend.clear_session()

    return val_mse


def run_hyperparameter_tuning(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lang: str = "es",
    save_dir: str = "outputs",
    n_trials: int = 20,
) -> Dict[str, Any]:
    if not HAS_OPTUNA:
        return {
            "model_name": model_name,
            "best_params": {},
            "best_mse": 0.0,
            "best_r2": 0.0,
            "top_3": [],
            "n_trials": 0,
            "optimization_history_plot": "",
            "param_importance_plot": "",
            "summary": _tr(lang, "optuna_missing"),
        }

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val).ravel()
    input_shape = X_train.shape[1]

    is_ae = "autoencoder" in model_name.lower()

    summary_parts: List[str] = []
    summary_parts.append(_tr(lang, "summary_intro"))
    summary_parts.append(f"  Modelo: {model_name}, Trials: {n_trials}")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(),
    )

    study.optimize(
        lambda trial: _objective(
            trial, model_name,
            X_train, y_train,
            X_val, y_val,
            input_shape, is_ae,
        ),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    if len(study.trials) == 0:
        return {
            "model_name": model_name,
            "best_params": {},
            "best_mse": 0.0,
            "best_r2": 0.0,
            "top_3": [],
            "n_trials": 0,
            "optimization_history_plot": "",
            "param_importance_plot": "",
            "summary": _tr(lang, "no_trials"),
        }

    best_params = study.best_params
    best_mse = float(study.best_value)

    best_model = _build_tuning_model(
        model_name, input_shape,
        best_params["learning_rate"],
        best_params["units"],
        best_params["dropout_rate"],
        best_params["num_layers"],
    )
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, restore_best_weights=True, verbose=0
    )
    if is_ae:
        best_model.fit(
            X_train, [y_train, X_train],
            validation_data=(X_val, [y_val, X_val]),
            epochs=50, batch_size=best_params["batch_size"],
            callbacks=[es], verbose=0,
        )
        y_pred_best = best_model.predict(X_val, verbose=0)[0].ravel()
    else:
        best_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50, batch_size=best_params["batch_size"],
            callbacks=[es], verbose=0,
        )
        y_pred_best = best_model.predict(X_val, verbose=0).ravel()
    best_r2 = float(r2_score(y_val, y_pred_best))
    keras.backend.clear_session()

    sorted_trials = sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
    )
    top_3: List[Dict[str, Any]] = []
    for i, trial in enumerate(sorted_trials[:3]):
        p = trial.params
        m = trial.value
        model_temp = _build_tuning_model(
            model_name, input_shape,
            p["learning_rate"], p["units"],
            p["dropout_rate"], p["num_layers"],
        )
        es2 = keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True, verbose=0
        )
        if is_ae:
            model_temp.fit(
                X_train, [y_train, X_train],
                validation_data=(X_val, [y_val, X_val]),
                epochs=30, batch_size=p["batch_size"],
                callbacks=[es2], verbose=0,
            )
            yp = model_temp.predict(X_val, verbose=0)[0].ravel()
        else:
            model_temp.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=30, batch_size=p["batch_size"],
                callbacks=[es2], verbose=0,
            )
            yp = model_temp.predict(X_val, verbose=0).ravel()
        r2t = float(r2_score(y_val, yp))
        top_3.append({"params": p, "mse": m, "r2": r2t})
        keras.backend.clear_session()

    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    trial_numbers = [t.number for t in study.trials if t.value is not None]
    trial_values = [t.value for t in study.trials if t.value is not None]
    ax_hist.plot(trial_numbers, trial_values, "o-", color="#1f77b4", alpha=0.7, markersize=4)
    ax_hist.set_xlabel(_tr(lang, "trial").split("/")[0].strip())
    ax_hist.set_ylabel(_tr(lang, "objective"))
    ax_hist.set_title(_tr(lang, "history_title"))
    ax_hist.grid(True, alpha=0.3)
    fig_hist.tight_layout()
    history_plot = _save_plot(fig_hist, save_path / "optimization_history.png")

    try:
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        importance = optuna.importance.get_param_importances(study)
        param_names = list(importance.keys())
        importances = list(importance.values())
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(param_names)))
        bars = ax_imp.barh(range(len(param_names)), importances, color=colors, edgecolor="black")
        ax_imp.set_yticks(range(len(param_names)))
        ax_imp.set_yticklabels([_tr(lang, p.replace("_rate", "").replace("_", "_")) if p.replace("_rate", "") in ["dropout", "learning", "num_layers"] else p for p in param_names])
        ax_imp.set_xlabel(_tr(lang, "importance"))
        ax_imp.set_title(_tr(lang, "importance_title"))
        ax_imp.invert_yaxis()
        ax_imp.grid(True, alpha=0.3, axis="x")
        fig_imp.tight_layout()
        importance_plot = _save_plot(fig_imp, save_path / "param_importance.png")
    except Exception:
        importance_plot = ""

    summary_parts.append(
        _tr(lang, "summary_best", mse=best_mse, r2=best_r2)
    )
    summary_parts.append(_tr(lang, "summary_trials", n=len(study.trials)))
    summary_parts.append(_tr(lang, "plots_saved", dir=str(save_path.resolve())))

    return {
        "model_name": model_name,
        "best_params": best_params,
        "best_mse": best_mse,
        "best_r2": best_r2,
        "top_3": top_3,
        "n_trials": len([t for t in study.trials if t.value is not None]),
        "optimization_history_plot": history_plot,
        "param_importance_plot": importance_plot,
        "summary": "\n".join(summary_parts),
    }
