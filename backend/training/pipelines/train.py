import io
import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

warnings.filterwarnings("ignore", category=FutureWarning)

_LANG: Dict[str, Dict[str, str]] = {
    "es": {
        "loss_history": "Historial de Pérdida - {name}",
        "comparison_title": "Comparación de Curvas de Pérdida",
        "epoch": "Época",
        "loss": "Pérdida",
        "val_loss": "Pérdida de Validación",
        "confusion_matrix_title": "Matriz de Confusión (Cuartiles) - {name}",
        "true_label": "Etiqueta Real",
        "predicted_label": "Etiqueta Predicha",
        "roc_title": "Curva ROC - {name}",
        "fpr": "Tasa de Falsos Positivos",
        "tpr": "Tasa de Verdaderos Positivos",
        "auc": "AUC = {auc:.3f}",
        "model": "Modelo",
        "model_type": "Tipo",
        "mse": "MSE",
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "R²",
        "time": "Tiempo (s)",
        "params": "Parámetros",
        "best_epoch": "Mejor Época",
        "best_model": "Mejor Modelo",
        "summary_intro": "=== Resultados de Entrenamiento ===",
        "best_model_text": "Mejor modelo: {name} (R² = {r2:.4f})",
        "comparison_table_title": "Tabla Comparativa de Modelos",
        "plots_saved": "Gráficos guardados en: {dir}",
        "models_evaluated": "Modelos evaluados: {count}",
        "scaler_saved": "Scaler guardado en: {path}",
        "no_models": "No se pudieron entrenar modelos exitosamente.",
        "training": "Entrenando {name}...",
        "done": "Completado en {time:.1f}s - MSE: {mse:.4f}, R²: {r2:.4f}",
        "q0": "Q1",
        "q1": "Q2",
        "q2": "Q3",
        "q3": "Q4",
        "below_median": "Bajo Mediana",
        "above_median": "Sobre Mediana",
    },
    "en": {
        "loss_history": "Loss History - {name}",
        "comparison_title": "Loss Curves Comparison",
        "epoch": "Epoch",
        "loss": "Loss",
        "val_loss": "Validation Loss",
        "confusion_matrix_title": "Confusion Matrix (Quartiles) - {name}",
        "true_label": "True Label",
        "predicted_label": "Predicted Label",
        "roc_title": "ROC Curve - {name}",
        "fpr": "False Positive Rate",
        "tpr": "True Positive Rate",
        "auc": "AUC = {auc:.3f}",
        "model": "Model",
        "model_type": "Type",
        "mse": "MSE",
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "R²",
        "time": "Time (s)",
        "params": "Parameters",
        "best_epoch": "Best Epoch",
        "best_model": "Best Model",
        "summary_intro": "=== Training Results ===",
        "best_model_text": "Best model: {name} (R² = {r2:.4f})",
        "comparison_table_title": "Model Comparison Table",
        "plots_saved": "Plots saved to: {dir}",
        "models_evaluated": "Models evaluated: {count}",
        "scaler_saved": "Scaler saved to: {path}",
        "no_models": "No models could be trained successfully.",
        "training": "Training {name}...",
        "done": "Completed in {time:.1f}s - MSE: {mse:.4f}, R²: {r2:.4f}",
        "q0": "Q1",
        "q1": "Q2",
        "q2": "Q3",
        "q3": "Q4",
        "below_median": "Below Median",
        "above_median": "Above Median",
    },
}

MODEL_DISPLAY_NAMES: Dict[str, Dict[str, str]] = {
    "mlp": {"es": "MLP", "en": "MLP"},
    "deep_mlp": {"es": "MLP Profundo", "en": "Deep MLP"},
    "residual_mlp": {"es": "MLP Residual", "en": "Residual MLP"},
    "cnn_lstm": {"es": "CNN-LSTM", "en": "CNN-LSTM"},
    "autoencoder_mlp": {"es": "MLP Autoencoder", "en": "Autoencoder MLP"},
}

QUARTILE_LABELS = ["q0", "q1", "q2", "q3"]


def _tr(lang: str, key: str, **fmt: Any) -> str:
    val = _LANG.get(lang, _LANG["es"]).get(key, key)
    return val.format(**fmt) if fmt else val


def _get_model_type(name: str, lang: str) -> str:
    name_lower = name.lower().replace("-", "_").replace(" ", "_")
    for key in MODEL_DISPLAY_NAMES:
        if key in name_lower:
            return MODEL_DISPLAY_NAMES[key][lang]
    return name


def _is_autoencoder(model: keras.Model) -> bool:
    return len(model.outputs) > 1


def _save_plot(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_loss_history(
    history: keras.callbacks.History,
    name: str,
    save_dir: Path,
    lang: str,
) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs_range = range(1, len(history.history["loss"]) + 1)
    ax.plot(epochs_range, history.history["loss"], label=_tr(lang, "loss"), color="#1f77b4")
    val_key = "val_loss"
    if val_key in history.history:
        ax.plot(epochs_range, history.history[val_key], label=_tr(lang, "val_loss"), color="#ff7f0e")
    ax.set_xlabel(_tr(lang, "epoch"))
    ax.set_ylabel(_tr(lang, "loss"))
    ax.set_title(_tr(lang, "loss_history", name=name))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_plot(fig, save_dir / f"loss_{name}.png")


def _plot_loss_comparison(
    histories: Dict[str, Dict[str, List[float]]],
    save_dir: Path,
    lang: str,
) -> str:
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    for (model_name, hist), color in zip(histories.items(), colors):
        epochs_range = range(1, len(hist["loss"]) + 1)
        ax.plot(epochs_range, hist["loss"], "--", color=color, alpha=0.5, label=f"{model_name} ({_tr(lang, 'loss')})")
        if "val_loss" in hist:
            ax.plot(epochs_range, hist["val_loss"], "-", color=color, label=f"{model_name} ({_tr(lang, 'val_loss')})")
    ax.set_xlabel(_tr(lang, "epoch"))
    ax.set_ylabel(_tr(lang, "loss"))
    ax.set_title(_tr(lang, "comparison_title"))
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_plot(fig, save_dir / "loss_comparison.png")


def _plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    save_dir: Path,
    lang: str,
) -> str:
    bins = np.percentile(y_true, [0, 25, 50, 75, 100])
    bins[-1] += 1e-8
    y_true_binned = np.digitize(y_true, bins[1:-1])
    y_pred_binned = np.digitize(y_pred, bins[1:-1])
    cm = confusion_matrix(y_true_binned, y_pred_binned, labels=[0, 1, 2, 3])

    labels = [_tr(lang, q) for q in QUARTILE_LABELS]
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel(_tr(lang, "predicted_label"))
    ax.set_ylabel(_tr(lang, "true_label"))
    ax.set_title(_tr(lang, "confusion_matrix_title", name=name))
    fig.tight_layout()
    return _save_plot(fig, save_dir / f"confusion_matrix_{name}.png")


def _plot_roc_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    name: str,
    save_dir: Path,
    lang: str,
) -> str:
    median = np.median(y_true)
    y_true_binary = (y_true >= median).astype(int)
    fpr, tpr, _ = roc_curve(y_true_binary, y_pred)
    auc = roc_auc_score(y_true_binary, y_pred)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=_tr(lang, "auc", auc=auc))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel(_tr(lang, "fpr"))
    ax.set_ylabel(_tr(lang, "tpr"))
    ax.set_title(_tr(lang, "roc_title", name=name))
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return _save_plot(fig, save_dir / f"roc_{name}.png")


def run_training(
    model_builders: Dict[str, Callable],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    early_stopping_patience: int = 10,
    lang: str = "es",
    save_dir: str = "outputs",
) -> Dict[str, Any]:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val).ravel()
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test).ravel()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    with open(str(save_path / "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    scaler_path = str(save_path / "scaler.pkl")

    input_shape = X_train.shape[1]

    individual_results: Dict[str, Dict[str, Any]] = {}
    histories: Dict[str, Dict[str, List[float]]] = {}
    predictions: Dict[str, np.ndarray] = {}
    comparison_rows: List[Dict[str, Any]] = []
    loss_curve_plots: List[str] = []
    summary_parts: List[str] = []
    summary_parts.append(_tr(lang, "summary_intro"))

    for model_name, builder in model_builders.items():
        model_type = _get_model_type(model_name, lang)
        summary_parts.append(_tr(lang, "training", name=model_name))

        try:
            model = builder(input_shape=input_shape)
            is_ae = _is_autoencoder(model)

            es_callback = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=0,
            )
            lr_callback = keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=early_stopping_patience // 2,
                min_lr=1e-6,
                verbose=0,
            )
            cp_callback = keras.callbacks.ModelCheckpoint(
                str(save_path / f"checkpoint_{model_name}.weights.h5"),
                monitor="val_loss",
                save_best_only=True,
                save_weights_only=True,
                verbose=0,
            )

            start = time.time()
            if is_ae:
                hist = model.fit(
                    X_train_scaled,
                    [y_train, X_train_scaled],
                    validation_data=(X_val_scaled, [y_val, X_val_scaled]),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[es_callback, lr_callback, cp_callback],
                    verbose=0,
                )
            else:
                hist = model.fit(
                    X_train_scaled,
                    y_train,
                    validation_data=(X_val_scaled, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[es_callback, lr_callback, cp_callback],
                    verbose=0,
                )
            elapsed = time.time() - start

            best_epoch = int(es_callback.stopped_epoch) - early_stopping_patience + 1 if es_callback.stopped_epoch > 0 else epochs
            best_epoch = max(best_epoch, 1)
            num_params = int(model.count_params())

            if is_ae:
                y_pred = model.predict(X_test_scaled, verbose=0)[0].ravel()
            else:
                y_pred = model.predict(X_test_scaled, verbose=0).ravel()

            mse = float(mean_squared_error(y_test, y_pred))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            predictions[model_name] = y_pred

            training_loss = [float(v) for v in hist.history.get("loss", [])]
            val_loss = [float(v) for v in hist.history.get("val_loss", [])]
            histories[model_name] = {"loss": training_loss, "val_loss": val_loss}

            individual_results[model_name] = {
                "history": {"loss": training_loss, "val_loss": val_loss},
                "metrics": {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2},
                "best_epoch": best_epoch,
                "training_time_sec": round(elapsed, 2),
            }

            comparison_rows.append({
                "model_name": model_name,
                "model_type": model_type,
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "training_time_sec": round(elapsed, 2),
                "params": num_params,
                "best_epoch": best_epoch,
            })

            loss_curve_plots.append(_plot_loss_history(hist, model_name, save_path, lang))

            print(f"  {model_name}: MSE={mse:.4f}, R²={r2:.4f}, time={elapsed:.1f}s")
            summary_parts.append(_tr(lang, "done", time=elapsed, mse=mse, r2=r2))

        except Exception as e:
            msg = f"  Error entrenando {model_name}: {str(e)}"
            summary_parts.append(msg)
            print(msg)
            continue

    if not individual_results:
        return {
            "comparison_table": [],
            "individual_results": {},
            "loss_curve_plots": [],
            "confusion_matrix_plot": "",
            "roc_curve_plot": "",
            "best_model_name": "",
            "best_model_path": "",
            "scaler_path": scaler_path,
            "summary": _tr(lang, "no_models"),
        }

    if len(histories) > 1:
        loss_curve_plots.append(_plot_loss_comparison(histories, save_path, lang))

    best_model_name = max(individual_results, key=lambda n: individual_results[n]["metrics"]["r2"])
    best_r2 = individual_results[best_model_name]["metrics"]["r2"]
    print(f"  Best model: {best_model_name} (R²={best_r2:.4f})")

    summary_parts.append("")
    summary_parts.append(_tr(lang, "best_model_text", name=best_model_name, r2=best_r2))
    summary_parts.append("")

    best_y_pred = predictions[best_model_name]
    cm_plot = _plot_confusion_matrix(y_test, best_y_pred, best_model_name, save_path, lang)
    roc_plot = _plot_roc_curve(y_test, best_y_pred, best_model_name, save_path, lang)

    best_checkpoint = save_path / f"checkpoint_{best_model_name}.weights.h5"
    best_model_path = str(save_path / "best_model.h5")
    if best_checkpoint.exists():
        best_model = model_builders[best_model_name](input_shape=input_shape)
        best_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae", "mse"],
        )
        best_model.load_weights(str(best_checkpoint))
        best_model.save(best_model_path)
    else:
        best_model = model_builders[best_model_name](input_shape=input_shape)
        best_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae", "mse"],
        )
        dummy = np.zeros((1, input_shape))
        best_model.predict(dummy, verbose=0)
        best_model.save(best_model_path)

    summary_parts.append(_tr(lang, "models_evaluated", count=len(individual_results)))
    summary_parts.append(_tr(lang, "plots_saved", dir=str(save_path.resolve())))
    summary_parts.append(_tr(lang, "scaler_saved", path=scaler_path))

    return {
        "comparison_table": comparison_rows,
        "individual_results": individual_results,
        "loss_curve_plots": loss_curve_plots,
        "confusion_matrix_plot": cm_plot,
        "roc_curve_plot": roc_plot,
        "best_model_name": best_model_name,
        "best_model_path": best_model_path,
        "scaler_path": scaler_path,
        "summary": "\n".join(summary_parts),
    }
