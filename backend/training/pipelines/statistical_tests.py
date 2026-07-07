import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from scipy.stats import shapiro, kruskal, friedmanchisquare, wilcoxon, ttest_rel
except ImportError:
    pass

try:
    import statsmodels.api as sm
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import het_breuschpagan
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

_LANG: Dict[str, Dict[str, str]] = {
    "es": {
        "shapiro_title": "Test de Shapiro-Wilk - {name}",
        "dw_title": "Test de Durbin-Watson - {name}",
        "bp_title": "Test de Breusch-Pagan - {name}",
        "friedman_title": "Test de Friedman",
        "kruskal_title": "Test de Kruskal-Wallis",
        "pairwise_title": "Pruebas Pareadas",
        "residual_hist": "Histograma de Residuales - {name}",
        "qq_title": "Gráfico Q-Q de Residuales - {name}",
        "residuals_fitted": "Residuales vs Valores Ajustados - {name}",
        "scale_location": "Scale-Location - {name}",
        "fitted_values": "Valores Ajustados",
        "residuals": "Residuales",
        "theoretical_quantiles": "Cuantiles Teóricos",
        "sample_quantiles": "Cuantiles de la Muestra",
        "density": "Densidad",
        "frequency": "Frecuencia",
        "std_residuals": "Residuales Estandarizados",
        "sqrt_std_residuals": "Raíz de |Residuales Estandarizados|",
        "normal": "Distribución Normal",
        "not_normal": "No Normal",
        "autocorrelation": "Autocorrelación",
        "no_autocorrelation": "Sin Autocorrelación",
        "heteroscedastic": "Heterocedástico",
        "no_heteroscedastic": "Homocedástico",
        "significant": "Diferencia Significativa",
        "not_significant": "Sin Diferencia Significativa",
        "summary_intro": "=== Resultados de Pruebas Estadísticas ===",
        "summary_shapiro": "Shapiro-Wilk: {result} (p={p:.4f})",
        "summary_dw": "Durbin-Watson: stat={stat:.4f} - {result}",
        "summary_bp": "Breusch-Pagan: {result} (p={p:.4f})",
        "summary_friedman": "Friedman: {result} (p={p:.4f})",
        "summary_kruskal": "Kruskal-Wallis: {result} (p={p:.4f})",
        "plots_saved": "Gráficos guardados en: {dir}",
        "no_statsmodels": "statsmodels no está instalado. Instálelo con: pip install statsmodels",
        "normal_interpretation": "Los residuales siguen una distribución normal (p={p:.4f} > 0.05).",
        "not_normal_interpretation": "Los residuales NO siguen una distribución normal (p={p:.4f} <= 0.05).",
        "dw_no_auto": "No se detecta autocorrelación significativa (stat={stat:.4f}, cerca de 2.0).",
        "dw_positive_auto": "Posible autocorrelación positiva (stat={stat:.4f} < 1.5).",
        "dw_negative_auto": "Posible autocorrelación negativa (stat={stat:.4f} > 2.5).",
        "bp_homo": "Varianza homogénea (p={p:.4f} > 0.05).",
        "bp_hetero": "Heterocedasticidad detectada (p={p:.4f} <= 0.05).",
        "friedman_sig": "Diferencias significativas entre modelos (p={p:.4f} <= 0.05).",
        "friedman_not_sig": "No hay diferencias significativas entre modelos (p={p:.4f} > 0.05).",
        "kruskal_sig": "Al menos un modelo difiere significativamente (p={p:.4f} <= 0.05).",
        "kruskal_not_sig": "No hay diferencias significativas entre modelos (p={p:.4f} > 0.05).",
        "pairwise_sig": "Diferencia significativa (p={p:.4f} <= 0.05)",
        "pairwise_not_sig": "Sin diferencia significativa (p={p:.4f} > 0.05)",
        "residual_plots_title": "Gráficos de Diagnóstico de Residuales",
        "vs": "vs",
    },
    "en": {
        "shapiro_title": "Shapiro-Wilk Test - {name}",
        "dw_title": "Durbin-Watson Test - {name}",
        "bp_title": "Breusch-Pagan Test - {name}",
        "friedman_title": "Friedman Test",
        "kruskal_title": "Kruskal-Wallis Test",
        "pairwise_title": "Pairwise Tests",
        "residual_hist": "Residual Histogram - {name}",
        "qq_title": "Q-Q Plot of Residuals - {name}",
        "residuals_fitted": "Residuals vs Fitted - {name}",
        "scale_location": "Scale-Location - {name}",
        "fitted_values": "Fitted Values",
        "residuals": "Residuals",
        "theoretical_quantiles": "Theoretical Quantiles",
        "sample_quantiles": "Sample Quantiles",
        "density": "Density",
        "frequency": "Frequency",
        "std_residuals": "Standardized Residuals",
        "sqrt_std_residuals": "Sqrt of |Standardized Residuals|",
        "normal": "Normal Distribution",
        "not_normal": "Not Normal",
        "autocorrelation": "Autocorrelation",
        "no_autocorrelation": "No Autocorrelation",
        "heteroscedastic": "Heteroscedastic",
        "no_heteroscedastic": "Homoscedastic",
        "significant": "Significant Difference",
        "not_significant": "No Significant Difference",
        "summary_intro": "=== Statistical Test Results ===",
        "summary_shapiro": "Shapiro-Wilk: {result} (p={p:.4f})",
        "summary_dw": "Durbin-Watson: stat={stat:.4f} - {result}",
        "summary_bp": "Breusch-Pagan: {result} (p={p:.4f})",
        "summary_friedman": "Friedman: {result} (p={p:.4f})",
        "summary_kruskal": "Kruskal-Wallis: {result} (p={p:.4f})",
        "plots_saved": "Plots saved to: {dir}",
        "no_statsmodels": "statsmodels is not installed. Install it with: pip install statsmodels",
        "normal_interpretation": "Residuals follow a normal distribution (p={p:.4f} > 0.05).",
        "not_normal_interpretation": "Residuals do NOT follow a normal distribution (p={p:.4f} <= 0.05).",
        "dw_no_auto": "No significant autocorrelation detected (stat={stat:.4f}, near 2.0).",
        "dw_positive_auto": "Possible positive autocorrelation (stat={stat:.4f} < 1.5).",
        "dw_negative_auto": "Possible negative autocorrelation (stat={stat:.4f} > 2.5).",
        "bp_homo": "Homogeneous variance (p={p:.4f} > 0.05).",
        "bp_hetero": "Heteroscedasticity detected (p={p:.4f} <= 0.05).",
        "friedman_sig": "Significant differences between models (p={p:.4f} <= 0.05).",
        "friedman_not_sig": "No significant differences between models (p={p:.4f} > 0.05).",
        "kruskal_sig": "At least one model differs significantly (p={p:.4f} <= 0.05).",
        "kruskal_not_sig": "No significant differences between models (p={p:.4f} > 0.05).",
        "pairwise_sig": "Significant difference (p={p:.4f} <= 0.05)",
        "pairwise_not_sig": "No significant difference (p={p:.4f} > 0.05)",
        "residual_plots_title": "Residual Diagnostic Plots",
        "vs": "vs",
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


def _plot_residual_diagnostics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residuals: np.ndarray,
    model_name: str,
    save_dir: Path,
    lang: str,
) -> List[str]:
    plots: List[str] = []

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax1 = axes[0, 0]
    ax1.hist(residuals, bins=50, edgecolor="black", alpha=0.7, color="#1f77b4", density=True)
    mu, std = np.mean(residuals), np.std(residuals)
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mu, std), "r-", lw=2, label=_tr(lang, "normal"))
    ax1.set_xlabel(_tr(lang, "residuals"))
    ax1.set_ylabel(_tr(lang, "density"))
    ax1.set_title(_tr(lang, "residual_hist", name=model_name))
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title(_tr(lang, "qq_title", name=model_name))
    ax2.set_xlabel(_tr(lang, "theoretical_quantiles"))
    ax2.set_ylabel(_tr(lang, "sample_quantiles"))
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.scatter(y_pred, residuals, alpha=0.5, s=10, color="#1f77b4")
    ax3.axhline(y=0, color="red", linestyle="--", lw=1)
    ax3.set_xlabel(_tr(lang, "fitted_values"))
    ax3.set_ylabel(_tr(lang, "residuals"))
    ax3.set_title(_tr(lang, "residuals_fitted", name=model_name))
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    std_residuals = residuals / (np.std(residuals) + 1e-10)
    ax4.scatter(y_pred, np.sqrt(np.abs(std_residuals)), alpha=0.5, s=10, color="#1f77b4")
    ax4.set_xlabel(_tr(lang, "fitted_values"))
    ax4.set_ylabel(_tr(lang, "sqrt_std_residuals"))
    ax4.set_title(_tr(lang, "scale_location", name=model_name))
    ax4.grid(True, alpha=0.3)

    fig.suptitle(_tr(lang, "residual_plots_title"), fontsize=14, y=1.02)
    fig.tight_layout()
    plots.append(_save_plot(fig, save_dir / f"residual_diagnostics_{model_name}.png"))

    return plots


def _run_shapiro_wilk(
    residuals: np.ndarray,
    lang: str,
) -> Dict[str, Any]:
    statistic, p_value = shapiro(residuals)
    statistic = float(statistic)
    p_value = float(p_value)
    normal = p_value > 0.05
    if normal:
        interpretation = _tr(lang, "normal_interpretation", p=p_value)
    else:
        interpretation = _tr(lang, "not_normal_interpretation", p=p_value)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "normal": normal,
        "interpretation": interpretation,
    }


def _run_durbin_watson(
    residuals: np.ndarray,
    lang: str,
) -> Dict[str, Any]:
    if not HAS_STATSMODELS:
        return {"statistic": 0.0, "autocorrelation": False, "interpretation": _tr(lang, "no_statsmodels")}
    statistic = float(durbin_watson(residuals))
    if statistic < 1.5:
        autocorrelation = True
        interpretation = _tr(lang, "dw_positive_auto", stat=statistic)
    elif statistic > 2.5:
        autocorrelation = True
        interpretation = _tr(lang, "dw_negative_auto", stat=statistic)
    else:
        autocorrelation = False
        interpretation = _tr(lang, "dw_no_auto", stat=statistic)
    return {
        "statistic": statistic,
        "autocorrelation": autocorrelation,
        "interpretation": interpretation,
    }


def _run_breusch_pagan(
    residuals: np.ndarray,
    y_pred: np.ndarray,
    lang: str,
) -> Dict[str, Any]:
    if not HAS_STATSMODELS:
        return {"statistic": 0.0, "p_value": 1.0, "heteroscedastic": False, "interpretation": _tr(lang, "no_statsmodels")}
    exog = sm.add_constant(y_pred)
    bp_stat, bp_p, _, _ = het_breuschpagan(residuals, exog)
    bp_stat = float(bp_stat)
    bp_p = float(bp_p)
    heteroscedastic = bp_p <= 0.05
    if heteroscedastic:
        interpretation = _tr(lang, "bp_hetero", p=bp_p)
    else:
        interpretation = _tr(lang, "bp_homo", p=bp_p)
    return {
        "statistic": bp_stat,
        "p_value": bp_p,
        "heteroscedastic": heteroscedastic,
        "interpretation": interpretation,
    }


def _run_friedman_test(
    model_errors: Dict[str, np.ndarray],
    lang: str,
) -> Dict[str, Any]:
    if len(model_errors) < 2:
        return {"statistic": 0.0, "p_value": 1.0, "significant_difference": False, "interpretation": "Se requieren al menos 2 modelos."}
    min_len = min(len(e) for e in model_errors.values())
    aligned = [e[:min_len] for e in model_errors.values()]
    statistic, p_value = friedmanchisquare(*aligned)
    statistic = float(statistic)
    p_value = float(p_value)
    significant = p_value <= 0.05
    if significant:
        interpretation = _tr(lang, "friedman_sig", p=p_value)
    else:
        interpretation = _tr(lang, "friedman_not_sig", p=p_value)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant_difference": significant,
        "interpretation": interpretation,
    }


def _run_kruskal_wallis(
    model_errors: Dict[str, np.ndarray],
    lang: str,
) -> Dict[str, Any]:
    if len(model_errors) < 2:
        return {"statistic": 0.0, "p_value": 1.0, "significant_difference": False, "interpretation": "Se requieren al menos 2 modelos."}
    statistic, p_value = kruskal(*model_errors.values())
    statistic = float(statistic)
    p_value = float(p_value)
    significant = p_value <= 0.05
    if significant:
        interpretation = _tr(lang, "kruskal_sig", p=p_value)
    else:
        interpretation = _tr(lang, "kruskal_not_sig", p=p_value)
    return {
        "statistic": statistic,
        "p_value": p_value,
        "significant_difference": significant,
        "interpretation": interpretation,
    }


def _run_pairwise_tests(
    model_errors: Dict[str, np.ndarray],
    lang: str,
) -> Dict[str, Any]:
    pairwise: Dict[str, Any] = {}
    model_names = list(model_errors.keys())
    if len(model_names) < 2:
        return pairwise

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name1, name2 = model_names[i], model_names[j]
            key = f"{name1}_vs_{name2}"
            e1, e2 = model_errors[name1], model_errors[name2]
            min_len = min(len(e1), len(e2))
            e1, e2 = e1[:min_len], e2[:min_len]

            try:
                t_stat, t_p = ttest_rel(e1, e2)
                t_stat = float(t_stat)
                t_p = float(t_p)
            except Exception:
                t_stat, t_p = 0.0, 1.0

            try:
                w_stat, w_p = wilcoxon(e1, e2)
                w_stat = float(w_stat)
                w_p = float(w_p)
            except Exception:
                w_stat, w_p = 0.0, 1.0

            significant = t_p <= 0.05 or w_p <= 0.05
            if significant:
                interpretation = _tr(lang, "pairwise_sig", p=min(t_p, w_p))
            else:
                interpretation = _tr(lang, "pairwise_not_sig", p=max(t_p, w_p))

            pairwise[key] = {
                "paired_t_test": {"statistic": t_stat, "p_value": t_p},
                "wilcoxon_signed_rank": {"statistic": w_stat, "p_value": w_p},
                "significant": significant,
                "interpretation": interpretation,
            }
    return pairwise


def _compute_errors(models_results: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        name: np.abs(y_true.ravel() - pred.ravel())
        for name, pred in models_results.items()
    }


def run_statistical_tests_on_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    lang: str = "es",
    save_dir: str = "outputs",
) -> Dict[str, Any]:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = y_true - y_pred

    summary_parts: List[str] = []
    summary_parts.append(_tr(lang, "summary_intro"))
    summary_parts.append(f"  Modelo: {model_name}")

    shapiro_result = _run_shapiro_wilk(residuals, lang)
    summary_parts.append(
        _tr(lang, "summary_shapiro",
            result=_tr(lang, "normal" if shapiro_result["normal"] else "not_normal"),
            p=shapiro_result["p_value"])
    )

    dw_result = _run_durbin_watson(residuals, lang)
    summary_parts.append(
        _tr(lang, "summary_dw",
            stat=dw_result["statistic"],
            result=_tr(lang, "autocorrelation" if dw_result["autocorrelation"] else "no_autocorrelation"))
    )

    bp_result = _run_breusch_pagan(residuals, y_pred, lang)
    summary_parts.append(
        _tr(lang, "summary_bp",
            result=_tr(lang, "heteroscedastic" if bp_result["heteroscedastic"] else "no_heteroscedastic"),
            p=bp_result["p_value"])
    )

    residual_plots = _plot_residual_diagnostics(y_true, y_pred, residuals, model_name, save_path, lang)

    summary_parts.append(_tr(lang, "plots_saved", dir=str(save_path.resolve())))

    return {
        "shapiro_wilk": shapiro_result,
        "durbin_watson": dw_result,
        "breusch_pagan": bp_result,
        "friedman_test": {"statistic": 0.0, "p_value": 1.0, "significant_difference": False, "interpretation": "No aplica (un solo modelo)."},
        "pairwise_tests": {},
        "residual_plots": residual_plots,
        "summary": "\n".join(summary_parts),
    }


def run_statistical_tests(
    models_results: Dict[str, np.ndarray],
    y_true: np.ndarray,
    lang: str = "es",
    save_dir: str = "outputs",
) -> Dict[str, Any]:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true).ravel()
    models_results = {
        name: np.asarray(pred).ravel()
        for name, pred in models_results.items()
    }

    summary_parts: List[str] = []
    summary_parts.append(_tr(lang, "summary_intro"))

    model_errors = _compute_errors(models_results, y_true)

    shapiro_results: Dict[str, Any] = {}
    dw_results: Dict[str, Any] = {}
    bp_results: Dict[str, Any] = {}
    all_residual_plots: List[str] = []

    for model_name, y_pred in models_results.items():
        residuals = y_true - y_pred

        shapiro_results[model_name] = _run_shapiro_wilk(residuals, lang)
        dw_results[model_name] = _run_durbin_watson(residuals, lang)
        bp_results[model_name] = _run_breusch_pagan(residuals, y_pred, lang)

        residual_plots = _plot_residual_diagnostics(
            y_true, y_pred, residuals, model_name, save_path, lang
        )
        all_residual_plots.extend(residual_plots)

        s = shapiro_results[model_name]
        summary_parts.append(
            f"  {model_name} - " +
            _tr(lang, "summary_shapiro",
                result=_tr(lang, "normal" if s["normal"] else "not_normal"),
                p=s["p_value"])
        )

    if len(models_results) >= 2:
        friedman_result = _run_friedman_test(model_errors, lang)
        summary_parts.append(
            _tr(lang, "summary_friedman",
                result=_tr(lang, "significant" if friedman_result["significant_difference"] else "not_significant"),
                p=friedman_result["p_value"])
        )

        kruskal_result = _run_kruskal_wallis(model_errors, lang)
        summary_parts.append(
            _tr(lang, "summary_kruskal",
                result=_tr(lang, "significant" if kruskal_result["significant_difference"] else "not_significant"),
                p=kruskal_result["p_value"])
        )

        pairwise_tests = _run_pairwise_tests(model_errors, lang)
    else:
        friedman_result = {"statistic": 0.0, "p_value": 1.0, "significant_difference": False, "interpretation": "Se requieren al menos 2 modelos."}
        kruskal_result = {"statistic": 0.0, "p_value": 1.0, "significant_difference": False, "interpretation": "Se requieren al menos 2 modelos."}
        pairwise_tests = {}

    summary_parts.append(_tr(lang, "plots_saved", dir=str(save_path.resolve())))

    first_model = list(models_results.keys())[0]
    return {
        "shapiro_wilk": shapiro_results,
        "durbin_watson": dw_results,
        "breusch_pagan": bp_results,
        "friedman_test": friedman_result,
        "pairwise_tests": pairwise_tests,
        "kruskal_wallis": kruskal_result,
        "residual_plots": all_residual_plots,
        "summary": "\n".join(summary_parts),
    }
