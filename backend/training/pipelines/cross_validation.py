import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tensorflow import keras

warnings.filterwarnings("ignore", category=FutureWarning)

_LANG: Dict[str, Dict[str, str]] = {
    "es": {
        "cv_title": "Validación Cruzada K-Fold - {name}",
        "fold_label": "Pliegue {n}",
        "fold": "Pliegue",
        "mse": "MSE",
        "r2": "R²",
        "mean_mse": "MSE Promedio",
        "std_mse": "Desv. Est. MSE",
        "mean_r2": "R² Promedio",
        "std_r2": "Desv. Est. R²",
        "boxplot_title": "Distribución de R² por Pliegue",
        "r2_score": "Puntuación R²",
        "training": "Entrenando pliegue {fold}/{total}...",
        "done": "Completado - MSE: {mse:.4f}, R²: {r2:.4f}",
        "summary_intro": "=== Resultados de Validación Cruzada ===",
        "summary_model": "Modelo: {name}, {n_folds} pliegues",
        "summary_r2": "R²: {mean:.4f} ± {std:.4f}",
        "summary_mse": "MSE: {mean:.4f} ± {std:.4f}",
        "plots_saved": "Gráfico guardado en: {path}",
        "no_folds": "No se pudieron completar pliegues.",
    },
    "en": {
        "cv_title": "K-Fold Cross Validation - {name}",
        "fold_label": "Fold {n}",
        "fold": "Fold",
        "mse": "MSE",
        "r2": "R²",
        "mean_mse": "Mean MSE",
        "std_mse": "MSE Std",
        "mean_r2": "Mean R²",
        "std_r2": "R² Std",
        "boxplot_title": "R² Distribution Across Folds",
        "r2_score": "R² Score",
        "training": "Training fold {fold}/{total}...",
        "done": "Completed - MSE: {mse:.4f}, R²: {r2:.4f}",
        "summary_intro": "=== Cross-Validation Results ===",
        "summary_model": "Model: {name}, {n_folds} folds",
        "summary_r2": "R²: {mean:.4f} ± {std:.4f}",
        "summary_mse": "MSE: {mean:.4f} ± {std:.4f}",
        "plots_saved": "Plot saved to: {path}",
        "no_folds": "No folds could be completed.",
    },
}

MODEL_DISPLAY_NAMES: Dict[str, Dict[str, str]] = {
    "mlp": {"es": "MLP", "en": "MLP"},
    "deep_mlp": {"es": "MLP Profundo", "en": "Deep MLP"},
    "residual_mlp": {"es": "MLP Residual", "en": "Residual MLP"},
    "cnn_lstm": {"es": "CNN-LSTM", "en": "CNN-LSTM"},
    "autoencoder_mlp": {"es": "MLP Autoencoder", "en": "Autoencoder MLP"},
}


def _tr(lang: str, key: str, **fmt: Any) -> str:
    val = _LANG.get(lang, _LANG["es"]).get(key, key)
    return val.format(**fmt) if fmt else val


def _get_model_display_name(builder: Callable, lang: str) -> str:
    name = getattr(builder, "__name__", str(builder)).lower().replace("build_", "")
    for key in MODEL_DISPLAY_NAMES:
        if key in name:
            return MODEL_DISPLAY_NAMES[key][lang]
    return name


def _is_autoencoder_model(builder: Callable, input_shape: int) -> bool:
    model = builder(input_shape=input_shape)
    is_ae = len(model.outputs) > 1
    keras.backend.clear_session()
    return is_ae


def _save_plot(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_boxplot(
    fold_results: List[Dict[str, Any]],
    model_name: str,
    save_dir: Path,
    lang: str,
) -> str:
    r2_scores = [r["r2"] for r in fold_results]
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(r2_scores, patch_artist=True, widths=0.5)
    for patch in bp["boxes"]:
        patch.set_facecolor("#4ECDC4")
        patch.set_alpha(0.7)
    for median_line in bp["medians"]:
        median_line.set_color("black")
        median_line.set_linewidth(2)

    ax.scatter(range(1, len(r2_scores) + 1), r2_scores, color="darkorange", s=60, zorder=5, label=_tr(lang, "fold_label", n=""))
    if len(r2_scores) > 1:
        ax.set_xticks(range(1, len(r2_scores) + 1))
        ax.set_xticklabels([_tr(lang, "fold_label", n=i + 1) for i in range(len(r2_scores))])
    ax.set_ylabel(_tr(lang, "r2_score"))
    ax.set_title(_tr(lang, "boxplot_title"))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return _save_plot(fig, save_dir / f"cv_boxplot_{model_name}.png")


def run_cross_validation(
    model_builder: Callable,
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    epochs: int = 50,
    batch_size: int = 32,
    lang: str = "es",
    save_dir: str = "outputs",
) -> Dict[str, Any]:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X)
    y = np.asarray(y).ravel()
    input_shape = X.shape[1]

    model_name = _get_model_display_name(model_builder, lang)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results: List[Dict[str, Any]] = []
    summary_parts: List[str] = []
    summary_parts.append(_tr(lang, "summary_intro"))
    summary_parts.append(_tr(lang, "summary_model", name=model_name, n_folds=n_folds))

    is_ae = _is_autoencoder_model(model_builder, input_shape)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        fold_num = fold_idx + 1
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        summary_parts.append(_tr(lang, "training", fold=fold_num, total=n_folds))

        try:
            model = model_builder(input_shape=input_shape)

            es_callback = keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=0,
            )

            if is_ae:
                model.fit(
                    X_train_fold,
                    [y_train_fold, X_train_fold],
                    validation_data=(X_val_fold, [y_val_fold, X_val_fold]),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[es_callback],
                    verbose=0,
                )
                y_pred = model.predict(X_val_fold, verbose=0)[0].ravel()
            else:
                model.fit(
                    X_train_fold,
                    y_train_fold,
                    validation_data=(X_val_fold, y_val_fold),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[es_callback],
                    verbose=0,
                )
                y_pred = model.predict(X_val_fold, verbose=0).ravel()

            mse = float(mean_squared_error(y_val_fold, y_pred))
            r2 = float(r2_score(y_val_fold, y_pred))

            fold_results.append({"fold": fold_num, "mse": mse, "r2": r2})
            summary_parts.append(_tr(lang, "done", mse=mse, r2=r2))

            keras.backend.clear_session()

        except Exception as e:
            summary_parts.append(f"  Error en pliegue {fold_num}: {str(e)}")
            keras.backend.clear_session()
            continue

    if not fold_results:
        return {
            "model_name": model_name,
            "n_folds": n_folds,
            "fold_results": [],
            "mean_r2": 0.0,
            "std_r2": 0.0,
            "mean_mse": 0.0,
            "std_mse": 0.0,
            "boxplot_path": "",
            "summary": _tr(lang, "no_folds"),
        }

    r2_values = [r["r2"] for r in fold_results]
    mse_values = [r["mse"] for r in fold_results]
    mean_r2 = float(np.mean(r2_values))
    std_r2 = float(np.std(r2_values))
    mean_mse = float(np.mean(mse_values))
    std_mse = float(np.std(mse_values))

    boxplot_path = _plot_boxplot(fold_results, model_name, save_path, lang)

    summary_parts.append("")
    summary_parts.append(_tr(lang, "mean_r2") + f": {mean_r2:.4f}")
    summary_parts.append(_tr(lang, "std_r2") + f": {std_r2:.4f}")
    summary_parts.append(_tr(lang, "mean_mse") + f": {mean_mse:.4f}")
    summary_parts.append(_tr(lang, "std_mse") + f": {std_mse:.4f}")
    summary_parts.append("")
    summary_parts.append(_tr(lang, "summary_r2", mean=mean_r2, std=std_r2))
    summary_parts.append(_tr(lang, "summary_mse", mean=mean_mse, std=std_mse))
    summary_parts.append(_tr(lang, "plots_saved", path=boxplot_path))

    return {
        "model_name": model_name,
        "n_folds": n_folds,
        "fold_results": fold_results,
        "mean_r2": mean_r2,
        "std_r2": std_r2,
        "mean_mse": mean_mse,
        "std_mse": std_mse,
        "boxplot_path": boxplot_path,
        "summary": "\n".join(summary_parts),
    }
