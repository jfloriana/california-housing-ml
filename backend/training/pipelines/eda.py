import io
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.datasets import fetch_california_housing

warnings.filterwarnings("ignore", category=FutureWarning)

_LANG: Dict[str, Dict[str, str]] = {
    "es": {
        "count": "Cuenta",
        "mean": "Media",
        "std": "Desv. Est.",
        "min": "Mín",
        "max": "Máx",
        "skewness": "Asimetría",
        "kurtosis": "Curtosis",
        "missing_count": "Valores Faltantes",
        "missing_pct": "% Faltante",
        "outliers": "Valores Atípicos",
        "outlier_count": "Cantidad",
        "outlier_pct": "% Atípicos",
        "duplicates": "Filas Duplicadas",
        "dataset_shape": "Dimensiones del Dataset",
        "data_types": "Tipos de Datos",
        "correlation_with_target": "Correlación con el Objetivo",
        "hist_title": "Histograma de {col}",
        "hist_all_title": "Histogramas de Variables",
        "box_title": "Diagrama de Caja - {col}",
        "box_all_title": "Diagramas de Caja",
        "corr_title": "Matriz de Correlación",
        "pair_title": "Pairplot - Top 4 Correlaciones con {target}",
        "dist_title": "Distribución de {target} con KDE",
        "qq_title": "Gráfico Q-Q de {target}",
        "qq_theoretical": "Cuantiles Teóricos",
        "qq_sample": "Cuantiles de la Muestra",
        "target": "ValorMedCas",
        "summary_intro": "Resumen del Análisis Exploratorio de Datos",
        "summary_shape": "El dataset tiene {rows} filas y {cols} columnas.",
        "summary_missing": "Valores faltantes: {count} ({pct:.2f}%)",
        "summary_no_missing": "No se encontraron valores faltantes.",
        "summary_outliers_iqr": "Método IQR - Se detectaron valores atípicos en {cols} variable(s).",
        "summary_outliers_zscore": "Método Z-score - Se detectaron valores atípicos en {cols} variable(s).",
        "summary_duplicates": "Se encontraron {count} filas duplicadas.",
        "summary_no_duplicates": "No se encontraron filas duplicadas.",
        "summary_removed_outliers": "Se eliminaron {count} filas con valores atípicos (método: {method}).",
        "summary_correlation": "Las variables más correlacionadas con {target}:",
        "plots_saved": "Gráficos guardados en: {dir}",
        "feature": "Variable",
        "theoretical": "Teórico",
        "sample": "Muestra",
        "density": "Densidad",
        "frequency": "Frecuencia",
    },
    "en": {
        "count": "Count",
        "mean": "Mean",
        "std": "Std",
        "min": "Min",
        "max": "Max",
        "skewness": "Skewness",
        "kurtosis": "Kurtosis",
        "missing_count": "Missing Values",
        "missing_pct": "% Missing",
        "outliers": "Outliers",
        "outlier_count": "Count",
        "outlier_pct": "% Outliers",
        "duplicates": "Duplicate Rows",
        "dataset_shape": "Dataset Shape",
        "data_types": "Data Types",
        "correlation_with_target": "Correlation with Target",
        "hist_title": "Histogram of {col}",
        "hist_all_title": "Variable Histograms",
        "box_title": "Box Plot - {col}",
        "box_all_title": "Box Plots",
        "corr_title": "Correlation Matrix",
        "pair_title": "Pairplot - Top 4 Correlations with {target}",
        "dist_title": "Distribution of {target} with KDE",
        "qq_title": "Q-Q Plot of {target}",
        "qq_theoretical": "Theoretical Quantiles",
        "qq_sample": "Sample Quantiles",
        "target": "MedHouseVal",
        "summary_intro": "Exploratory Data Analysis Summary",
        "summary_shape": "Dataset has {rows} rows and {cols} columns.",
        "summary_missing": "Missing values: {count} ({pct:.2f}%)",
        "summary_no_missing": "No missing values found.",
        "summary_outliers_iqr": "IQR method - Outliers detected in {cols} variable(s).",
        "summary_outliers_zscore": "Z-score method - Outliers detected in {cols} variable(s).",
        "summary_duplicates": "Found {count} duplicate rows.",
        "summary_no_duplicates": "No duplicate rows found.",
        "summary_removed_outliers": "Removed {count} rows with outliers (method: {method}).",
        "summary_correlation": "Top features correlated with {target}:",
        "plots_saved": "Plots saved to: {dir}",
        "feature": "Feature",
        "theoretical": "Theoretical",
        "sample": "Sample",
        "density": "Density",
        "frequency": "Frequency",
    },
}


def _tr(lang: str, key: str, **fmt: Any) -> str:
    val = _LANG.get(lang, _LANG["es"]).get(key, key)
    return val.format(**fmt) if fmt else val


FEATURES = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms",
    "Population", "AveOccup", "Latitude", "Longitude",
]
TARGET = "MedHouseVal"
ALL_COLS = FEATURES + [TARGET]


def _detect_outliers_iqr(df: pd.DataFrame, cols: List[str]) -> Dict[str, int]:
    outliers: Dict[str, int] = {}
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
    return outliers


def _detect_outliers_zscore(df: pd.DataFrame, cols: List[str]) -> Dict[str, int]:
    outliers: Dict[str, int] = {}
    for col in cols:
        z = np.abs(stats.zscore(df[col].dropna()))
        outliers[col] = int((z > 3).sum())
    return outliers


def _save_plot(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _plot_histograms(
    df: pd.DataFrame,
    cols: List[str],
    save_dir: Path,
    lang: str,
) -> List[str]:
    files: List[str] = []
    n = len(cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        df[col].hist(bins=50, ax=axes[i], edgecolor="black", alpha=0.7)
        axes[i].set_title(_tr(lang, "hist_title", col=col))
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(_tr(lang, "frequency"))
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(_tr(lang, "hist_all_title"), fontsize=16, y=1.02)
    fig.tight_layout()
    files.append(_save_plot(fig, save_dir / "histograms_grid.png"))

    for col in cols:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        df[col].hist(bins=50, ax=ax2, edgecolor="black", alpha=0.7)
        ax2.set_title(_tr(lang, "hist_title", col=col))
        ax2.set_xlabel(col)
        ax2.set_ylabel(_tr(lang, "frequency"))
        fig2.tight_layout()
        files.append(_save_plot(fig2, save_dir / f"hist_{col}.png"))

    return files


def _plot_boxplots(
    df: pd.DataFrame,
    cols: List[str],
    save_dir: Path,
    lang: str,
) -> List[str]:
    files: List[str] = []
    n = len(cols)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
    for i, col in enumerate(cols):
        df.boxplot(column=col, ax=axes[i])
        axes[i].set_title(_tr(lang, "box_title", col=col))
        axes[i].set_ylabel(col)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(_tr(lang, "box_all_title"), fontsize=16, y=1.02)
    fig.tight_layout()
    files.append(_save_plot(fig, save_dir / "boxplots_grid.png"))

    for col in cols:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        df.boxplot(column=col, ax=ax2)
        ax2.set_title(_tr(lang, "box_title", col=col))
        ax2.set_ylabel(col)
        fig2.tight_layout()
        files.append(_save_plot(fig2, save_dir / f"boxplot_{col}.png"))

    return files


def _plot_correlation_matrix(
    df: pd.DataFrame,
    save_dir: Path,
    lang: str,
) -> Tuple[pd.DataFrame, str]:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, square=True, linewidths=0.5, ax=ax,
    )
    ax.set_title(_tr(lang, "corr_title"), fontsize=14)
    fig.tight_layout()
    path = _save_plot(fig, save_dir / "correlation_matrix.png")
    return corr, path


def _plot_pairplot(
    df: pd.DataFrame,
    top_cols: List[str],
    target: str,
    save_dir: Path,
    lang: str,
) -> str:
    vars_ = top_cols + [target]
    g = sns.pairplot(df[vars_], diag_kind="kde", corner=True)
    g.fig.suptitle(
        _tr(lang, "pair_title", target=_tr(lang, "target")),
        fontsize=14, y=1.02,
    )
    g.fig.tight_layout()
    path = _save_plot(g.fig, save_dir / "pairplot_top4.png")
    return path


def _plot_target_distribution(
    target_series: pd.Series,
    target_name: str,
    save_dir: Path,
    lang: str,
) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(target_series, kde=True, bins=50, ax=ax, edgecolor="black")
    ax.set_title(_tr(lang, "dist_title", target=_tr(lang, "target")))
    ax.set_xlabel(target_name)
    ax.set_ylabel(_tr(lang, "density"))
    fig.tight_layout()
    return _save_plot(fig, save_dir / "target_distribution.png")


def _plot_qq(
    target_series: pd.Series,
    target_name: str,
    save_dir: Path,
    lang: str,
) -> str:
    fig, ax = plt.subplots(figsize=(8, 8))
    stats.probplot(target_series, dist="norm", plot=ax)
    ax.set_title(_tr(lang, "qq_title", target=_tr(lang, "target")))
    ax.set_xlabel(_tr(lang, "qq_theoretical"))
    ax.set_ylabel(_tr(lang, "qq_sample"))
    fig.tight_layout()
    return _save_plot(fig, save_dir / "qq_plot.png")


def run_eda(
    save_plots_dir: str = "plots",
    lang: str = "es",
    remove_outliers: bool = False,
    outlier_method: str = "iqr",
) -> Dict[str, Any]:
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    df.columns = ALL_COLS

    initial_shape = df.shape
    plot_files: List[str] = []
    summary_parts: List[str] = []
    save_dir = Path(save_plots_dir)

    # ── Missing Values ──────────────────────────────────────────────
    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(2)
    missing_df = pd.DataFrame({
        _tr(lang, "missing_count"): missing_count,
        _tr(lang, "missing_pct"): missing_pct,
    })
    missing_df.index.name = _tr(lang, "feature")
    total_missing = int(missing_count.sum())
    if total_missing > 0:
        summary_parts.append(
            _tr(lang, "summary_missing", count=total_missing, pct=missing_pct.mean())
        )
    else:
        summary_parts.append(_tr(lang, "summary_no_missing"))

    # ── Duplicates ──────────────────────────────────────────────────
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        summary_parts.append(_tr(lang, "summary_duplicates", count=dup_count))
    else:
        summary_parts.append(_tr(lang, "summary_no_duplicates"))

    # ── Outliers ────────────────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in num_cols if c in FEATURES]

    outliers_iqr = _detect_outliers_iqr(df, numeric_cols)
    outliers_zscore = _detect_outliers_zscore(df, numeric_cols)

    cols_with_iqr = sum(1 for v in outliers_iqr.values() if v > 0)
    cols_with_z = sum(1 for v in outliers_zscore.values() if v > 0)
    summary_parts.append(
        _tr(lang, "summary_outliers_iqr", cols=cols_with_iqr)
    )
    summary_parts.append(
        _tr(lang, "summary_outliers_zscore", cols=cols_with_z)
    )

    # ── Remove Outliers (optional) ──────────────────────────────────
    if remove_outliers:
        mask = pd.Series(True, index=df.index)
        if outlier_method == "iqr":
            for col in numeric_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask &= (df[col] >= lower) & (df[col] <= upper)
        elif outlier_method == "zscore":
            for col in numeric_cols:
                z = np.abs(stats.zscore(df[col]))
                mask &= z <= 3
        removed = int((~mask).sum())
        df = df[mask].reset_index(drop=True)
        summary_parts.append(
            _tr(lang, "summary_removed_outliers", count=removed, method=outlier_method)
        )

    # ── Descriptive Statistics ──────────────────────────────────────
    stats_df = df[ALL_COLS].describe(percentiles=[0.25, 0.5, 0.75]).T
    stats_df["skewness"] = df[ALL_COLS].skew()
    stats_df["kurtosis"] = df[ALL_COLS].kurtosis()
    stats_df = stats_df[
        ["count", "mean", "std", "min", "25%", "50%", "75%", "max", "skewness", "kurtosis"]
    ].round(4)
    stats_df.columns = [
        _tr(lang, "count"), _tr(lang, "mean"), _tr(lang, "std"),
        _tr(lang, "min"), "25%", "50%", "75%", _tr(lang, "max"),
        _tr(lang, "skewness"), _tr(lang, "kurtosis"),
    ]

    # ── Correlation ─────────────────────────────────────────────────
    corr_matrix, corr_plot = _plot_correlation_matrix(df, save_dir, lang)
    plot_files.append(corr_plot)

    corr_with_target = corr_matrix[TARGET].drop(TARGET).sort_values(ascending=False)

    summary_parts.append(
        _tr(lang, "summary_correlation", target=_tr(lang, "target"))
    )
    for col, val in corr_with_target.head(4).items():
        summary_parts.append(f"  {col}: {val:.4f}")

    # ── Top 4 correlated features for pairplot ──────────────────────
    top4 = corr_with_target.head(4).index.tolist()
    pairplot_path = _plot_pairplot(df, top4, TARGET, save_dir, lang)
    plot_files.append(pairplot_path)

    # ── Histograms ──────────────────────────────────────────────────
    plot_files.extend(_plot_histograms(df, ALL_COLS, save_dir, lang))

    # ── Boxplots ────────────────────────────────────────────────────
    plot_files.extend(_plot_boxplots(df, FEATURES, save_dir, lang))

    # ── Target distribution ─────────────────────────────────────────
    plot_files.append(
        _plot_target_distribution(df[TARGET], TARGET, save_dir, lang)
    )

    # ── Q-Q plot ────────────────────────────────────────────────────
    plot_files.append(_plot_qq(df[TARGET], TARGET, save_dir, lang))

    # ── Data types ──────────────────────────────────────────────────
    dtypes_df = df.dtypes.reset_index()
    dtypes_df.columns = [_tr(lang, "feature"), _tr(lang, "data_types")]

    # ── Summary ─────────────────────────────────────────────────────
    summary_parts.insert(
        0,
        _tr(lang, "summary_shape", rows=initial_shape[0], cols=initial_shape[1]),
    )
    summary_parts.append(
        _tr(lang, "plots_saved", dir=str(save_dir.resolve()))
    )

    return {
        "descriptive_stats": stats_df,
        "missing_values": missing_df,
        "outliers_iqr": outliers_iqr,
        "outliers_zscore": outliers_zscore,
        "correlation_matrix": corr_matrix,
        "correlation_with_target": corr_with_target,
        "dataset_shape": initial_shape,
        "data_types": dtypes_df,
        "plot_files": plot_files,
        "summary": "\n".join(summary_parts),
    }
