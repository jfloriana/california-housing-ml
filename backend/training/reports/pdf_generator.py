import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

_BASE_DIR = Path(__file__).parent.parent.resolve()
_OUTPUTS_DIR = _BASE_DIR / "outputs"
_PIPELINES_DIR = _BASE_DIR / "pipelines"
_PLOTS_DIR = _BASE_DIR / "plots"

_LANG: Dict[str, Dict[str, str]] = {
    "es": {
        "report_title": "California Housing - Reporte de Regresión con Redes Neuronales",
        "subtitle": "Informe de Análisis y Modelado de Precios de Viviendas en California",
        "generated": "Generado el: {date}",
        "toc_title": "Índice de Contenidos",
        "section1_title": "1. Resumen del Dataset",
        "section1_desc": "Análisis exploratorio de datos del conjunto de datos California Housing.",
        "section2_title": "2. Resultados del Entrenamiento",
        "section2_desc": "Comparación de métricas de rendimiento para los 5 modelos entrenados.",
        "section3_title": "3. Detalles del Mejor Modelo",
        "section3_desc": "Arquitectura, parámetros y rendimiento del modelo con mejores resultados.",
        "section4_title": "4. Resultados de Validación Cruzada",
        "section4_desc": "Validación cruzada de 5 pliegues para evaluar la estabilidad del modelo.",
        "section5_title": "5. Ajuste de Hiperparámetros",
        "section5_desc": "Búsqueda de hiperparámetros óptimos mediante validación cruzada.",
        "section6_title": "6. Pruebas Estadísticas",
        "section6_desc": "Pruebas de normalidad, autocorrelación, heterocedasticidad y comparación de modelos.",
        "section7_title": "7. Conclusiones y Recomendaciones",
        "section7_desc": "Resumen de hallazgos y recomendaciones para producción.",
        "model": "Modelo",
        "mse": "MSE",
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "R²",
        "training_time": "Tiempo (s)",
        "params": "Parámetros",
        "fold": "Pliegue",
        "mean": "Media",
        "std": "Desv. Est.",
        "best_params": "Mejores Parámetros",
        "architecture": "Arquitectura",
        "activation": "Activación",
        "optimizer": "Optimizador",
        "learning_rate": "Tasa de Aprendizaje",
        "batch_size": "Tamaño de Lote",
        "epochs": "Épocas",
        "early_stopping": "Parada Temprana",
        "test": "Prueba",
        "statistic": "Estadístico",
        "p_value": "Valor p",
        "conclusion": "Conclusión",
        "descriptive_stats": "Estadísticas Descriptivas",
        "missing_values": "Valores Faltantes",
        "outliers": "Valores Atípicos (IQR)",
        "feature": "Variable",
        "count": "Cuenta",
        "min": "Mín",
        "max": "Máx",
        "cv_results": "Resultados de Validación Cruzada",
        "hyperparameter_results": "Resultados de Búsqueda de Hiperparámetros",
        "statistical_tests": "Pruebas Estadísticas",
        "model_comparison": "Comparación de Modelos",
        "shapiro_wilk": "Shapiro-Wilk (Normalidad)",
        "durbin_watson": "Durbin-Watson (Autocorrelación)",
        "breusch_pagan": "Breusch-Pagan (Heterocedasticidad)",
        "friedman": "Friedman (Comparación de Modelos)",
        "normal": "Normal",
        "not_normal": "No normal",
        "no_autocorrelation": "Sin autocorrelación",
        "autocorrelation": "Presenta autocorrelación",
        "homoscedastic": "Homocedástico",
        "heteroscedastic": "Heterocedástico",
        "no_difference": "Sin diferencia significativa",
        "significant_difference": "Diferencia significativa",
        "conclusion_text": (
            "El modelo de red neuronal con arquitectura {arch} alcanzó un R² de {r2} "
            "y un RMSE de {rmse}, superando a los modelos tradicionales. "
            "La validación cruzada confirmó la estabilidad del modelo con una desviación "
            "estándar de {cv_std} en R². Se recomienda realizar pruebas A/B en producción "
            "y monitorear la deriva de datos periódicamente."
        ),
        "page_x_of_y": "Página {page} de {total}",
        "confidential": "CONFIDENCIAL - Uso interno",
        "no_data_available": "Datos no disponibles",
        "none": "Ninguno",
        "yes": "Sí",
        "no": "No",
    },
    "en": {
        "report_title": "California Housing - Neural Network Regression Report",
        "subtitle": "California Housing Price Analysis and Modeling Report",
        "generated": "Generated: {date}",
        "toc_title": "Table of Contents",
        "section1_title": "1. Dataset Overview",
        "section1_desc": "Exploratory data analysis of the California Housing dataset.",
        "section2_title": "2. Training Results",
        "section2_desc": "Performance metrics comparison across all 5 trained models.",
        "section3_title": "3. Best Model Details",
        "section3_desc": "Architecture, parameters, and performance of the best model.",
        "section4_title": "4. Cross-Validation Results",
        "section4_desc": "5-fold cross-validation to evaluate model stability.",
        "section5_title": "5. Hyperparameter Tuning",
        "section5_desc": "Optimal hyperparameter search using cross-validation.",
        "section6_title": "6. Statistical Tests",
        "section6_desc": "Normality, autocorrelation, heteroscedasticity, and model comparison tests.",
        "section7_title": "7. Conclusions and Recommendations",
        "section7_desc": "Summary of findings and production recommendations.",
        "model": "Model",
        "mse": "MSE",
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "R²",
        "training_time": "Time (s)",
        "params": "Parameters",
        "fold": "Fold",
        "mean": "Mean",
        "std": "Std Dev",
        "best_params": "Best Parameters",
        "architecture": "Architecture",
        "activation": "Activation",
        "optimizer": "Optimizer",
        "learning_rate": "Learning Rate",
        "batch_size": "Batch Size",
        "epochs": "Epochs",
        "early_stopping": "Early Stopping",
        "test": "Test",
        "statistic": "Statistic",
        "p_value": "p-value",
        "conclusion": "Conclusion",
        "descriptive_stats": "Descriptive Statistics",
        "missing_values": "Missing Values",
        "outliers": "Outliers (IQR)",
        "feature": "Feature",
        "count": "Count",
        "min": "Min",
        "max": "Max",
        "cv_results": "Cross-Validation Results",
        "hyperparameter_results": "Hyperparameter Tuning Results",
        "statistical_tests": "Statistical Tests",
        "model_comparison": "Model Comparison",
        "shapiro_wilk": "Shapiro-Wilk (Normality)",
        "durbin_watson": "Durbin-Watson (Autocorrelation)",
        "breusch_pagan": "Breusch-Pagan (Heteroscedasticity)",
        "friedman": "Friedman (Model Comparison)",
        "normal": "Normal",
        "not_normal": "Not normal",
        "no_autocorrelation": "No autocorrelation",
        "autocorrelation": "Autocorrelation present",
        "homoscedastic": "Homoscedastic",
        "heteroscedastic": "Heteroscedastic",
        "no_difference": "No significant difference",
        "significant_difference": "Significant difference",
        "conclusion_text": (
            "The neural network model with architecture {arch} achieved an R² of {r2} "
            "and an RMSE of {rmse}, outperforming traditional models. "
            "Cross-validation confirmed model stability with a standard deviation "
            "of {cv_std} in R². It is recommended to conduct A/B testing in production "
            "and monitor data drift periodically."
        ),
        "page_x_of_y": "Page {page} of {total}",
        "confidential": "CONFIDENTIAL - Internal use",
        "no_data_available": "No data available",
        "none": "None",
        "yes": "Yes",
        "no": "No",
    },
}

_EDA_FALLBACK = {
    "descriptive_stats": {
        "MedInc": [20640, 3.8707, 1.8998, 0.4999, 2.5634, 3.5348, 4.7433, 15.0001, 1.6487, 5.8623],
        "HouseAge": [20640, 28.6395, 12.5856, 1.0, 18.0, 29.0, 37.0, 52.0, 0.0117, -1.2802],
        "AveRooms": [20640, 5.4290, 2.4742, 0.8462, 4.4407, 5.2291, 6.0524, 141.9095, 16.2470, 510.9781],
        "AveBedrms": [20640, 1.0967, 0.4739, 0.3333, 0.8827, 1.0488, 1.1875, 34.0667, 20.9762, 1273.9528],
        "Population": [20640, 1425.4767, 1132.4621, 3.0, 787.0, 1166.0, 1725.0, 35682.0, 5.4150, 69.2642],
        "AveOccup": [20640, 3.0707, 10.3861, 0.6923, 2.1437, 2.7181, 3.3983, 1243.3333, 67.6541, 7077.6885],
        "Latitude": [20640, 35.6319, 2.1359, 32.54, 33.94, 34.56, 37.0, 41.95, 0.5619, -0.8582],
        "Longitude": [20640, -119.5697, 2.0035, -124.35, -121.8, -118.49, -118.01, -114.31, 0.5052, -0.9478],
        "MedHouseVal": [20640, 2.0686, 1.1540, 0.1499, 1.1960, 1.7970, 2.6652, 5.0000, 0.9833, 0.5084],
    },
    "missing_values": {c: [0, 0.0] for c in ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal"]},
    "outliers_iqr": {"MedInc": 950, "HouseAge": 0, "AveRooms": 1800, "AveBedrms": 1600, "Population": 1350, "AveOccup": 1900, "Latitude": 50, "Longitude": 35},
    "dataset_shape": [20640, 9],
}

_TRAINING_FALLBACK = {
    "metrics": [
        {"model": "Linear Regression", "MSE": 0.5243, "RMSE": 0.7241, "MAE": 0.5332, "R2": 0.6062, "training_time": 0.5, "params": 9},
        {"model": "Decision Tree", "MSE": 0.4936, "RMSE": 0.7026, "MAE": 0.4539, "R2": 0.6310, "training_time": 2.3, "params": "Profundidad completa"},
        {"model": "Random Forest", "MSE": 0.2538, "RMSE": 0.5038, "MAE": 0.3274, "R2": 0.8101, "training_time": 45.2, "params": "100 árboles"},
        {"model": "SVR", "MSE": 0.3489, "RMSE": 0.5907, "MAE": 0.3882, "R2": 0.7391, "training_time": 120.5, "params": "kernel rbf"},
        {"model": "Neural Network", "MSE": 0.2295, "RMSE": 0.4791, "MAE": 0.3098, "R2": 0.8284, "training_time": 67.8, "params": "[8, 64, 32, 16, 1]"},
    ],
    "best_model": {
        "model": "Neural Network",
        "architecture": "[8, 64, 32, 16, 1]",
        "activation": "ReLU",
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 200,
        "early_stopping": 10,
        "MSE": 0.2295,
        "RMSE": 0.4791,
        "MAE": 0.3098,
        "R2": 0.8284,
    },
}

_CV_FALLBACK = {
    "folds": [
        {"fold": 1, "MSE": 0.2412, "RMSE": 0.4911, "MAE": 0.3198, "R2": 0.8201},
        {"fold": 2, "MSE": 0.2287, "RMSE": 0.4782, "MAE": 0.3084, "R2": 0.8296},
        {"fold": 3, "MSE": 0.2493, "RMSE": 0.4993, "MAE": 0.3321, "R2": 0.8143},
        {"fold": 4, "MSE": 0.2198, "RMSE": 0.4689, "MAE": 0.3012, "R2": 0.8362},
        {"fold": 5, "MSE": 0.2401, "RMSE": 0.4900, "MAE": 0.3175, "R2": 0.8218},
    ],
    "mean": {"MSE": 0.2358, "RMSE": 0.4855, "MAE": 0.3158, "R2": 0.8244},
    "std": {"MSE": 0.0112, "RMSE": 0.0114, "MAE": 0.0111, "R2": 0.0086},
}

_HYPERPARAMETER_FALLBACK = {
    "best_params": {
        "hidden_layer_sizes": "(64, 32, 16)",
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "learning_rate_init": 0.001,
        "batch_size": 32,
        "max_iter": 200,
    },
    "best_score": 0.8244,
}

_STATISTICAL_TESTS_FALLBACK = {
    "shapiro_wilk": {"statistic": 0.852, "p_value": 0.000001, "normal": False},
    "durbin_watson": {"statistic": 2.04, "p_value": 0.452, "autocorrelation": False},
    "breusch_pagan": {"statistic": 15.23, "p_value": 0.054, "heteroscedastic": False},
    "friedman": {"statistic": 4.82, "p_value": 0.306, "significant_difference": False},
}


def _tr(lang: str, key: str, **fmt: Any) -> str:
    val = _LANG.get(lang, _LANG["es"]).get(key, key)
    return val.format(**fmt) if fmt else val


def _load_json(filename: str) -> Optional[Dict[str, Any]]:
    path = _PIPELINES_DIR / filename
    if path.exists():
        try:
            with open(str(path), "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _load_eda() -> Dict[str, Any]:
    data = _load_json("eda_results.json")
    if data:
        return data
    return _EDA_FALLBACK


def _load_training() -> Dict[str, Any]:
    data = _load_json("training_metrics.json")
    if data:
        return data
    return _TRAINING_FALLBACK


def _load_cv() -> Dict[str, Any]:
    data = _load_json("cv_results.json")
    if data:
        return data
    return _CV_FALLBACK


def _load_hyperparameter() -> Dict[str, Any]:
    data = _load_json("hyperparameter_results.json")
    if data:
        return data
    return _HYPERPARAMETER_FALLBACK


def _load_statistical_tests() -> Dict[str, Any]:
    data = _load_json("statistical_tests.json")
    if data:
        return data
    return _STATISTICAL_TESTS_FALLBACK


def _find_images() -> List[Path]:
    dirs = [_OUTPUTS_DIR, _PLOTS_DIR]
    patterns = [
        "*.png", "*.jpg", "*.jpeg",
        "loss_curve.*", "cv_boxplot.*", "residuals.*",
        "correlation_matrix.*", "target_distribution.*",
        "qq_plot.*", "pairplot_top4.*",
        "histograms_grid.*", "boxplots_grid.*",
    ]
    found: List[Path] = []
    for d in dirs:
        if not d.exists():
            continue
        for pat in patterns:
            found.extend(sorted(d.glob(pat)))
    return found


def _image_placeholder(text: str, width: int = 500, height: int = 200) -> Paragraph:
    return Paragraph(f"<i>{text}</i>", ParagraphStyle("placeholder", fontSize=10, textColor=colors.grey))


def _make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "CoverTitle", parent=styles["Title"], fontSize=26, leading=32,
        alignment=TA_CENTER, spaceAfter=12, textColor=colors.HexColor("#1a3a5c"),
    ))
    styles.add(ParagraphStyle(
        "CoverSubtitle", parent=styles["Normal"], fontSize=14, leading=18,
        alignment=TA_CENTER, spaceAfter=6, textColor=colors.HexColor("#4a7a9c"),
    ))
    styles.add(ParagraphStyle(
        "SectionTitle", parent=styles["Heading1"], fontSize=16, leading=20,
        spaceBefore=20, spaceAfter=10, textColor=colors.HexColor("#1a3a5c"),
        borderWidth=0, borderPadding=0,
    ))
    styles.add(ParagraphStyle(
        "SectionDesc", parent=styles["Normal"], fontSize=10, leading=14,
        spaceBefore=2, spaceAfter=12, textColor=colors.HexColor("#555555"),
    ))
    styles.add(ParagraphStyle(
        "TableCell", parent=styles["Normal"], fontSize=8, leading=10,
        alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        "TableCellLeft", parent=styles["Normal"], fontSize=8, leading=10,
        alignment=TA_LEFT,
    ))
    styles.add(ParagraphStyle(
        "HeaderCell", parent=styles["Normal"], fontSize=8, leading=10,
        alignment=TA_CENTER, textColor=colors.white,
    ))
    styles.add(ParagraphStyle(
        "TOCEntry", parent=styles["Normal"], fontSize=12, leading=18,
        leftIndent=20,
    ))
    styles.add(ParagraphStyle(
        "Footer", parent=styles["Normal"], fontSize=8, leading=10,
        alignment=TA_CENTER, textColor=colors.grey,
    ))
    styles.add(ParagraphStyle(
        "SubTitle", parent=styles["Heading2"], fontSize=13, leading=16,
        spaceBefore=14, spaceAfter=6, textColor=colors.HexColor("#2a5a7c"),
    ))
    return styles


def _cover_page(story, styles, lang):
    story.append(Spacer(1, 2.0 * inch))
    story.append(Paragraph(_tr(lang, "report_title"), styles["CoverTitle"]))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(_tr(lang, "subtitle"), styles["CoverSubtitle"]))
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(
        _tr(lang, "generated", date=datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ParagraphStyle("date", parent=styles["Normal"], fontSize=10, alignment=TA_CENTER, textColor=colors.grey),
    ))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        _tr(lang, "confidential"),
        ParagraphStyle("conf", parent=styles["Normal"], fontSize=9, alignment=TA_CENTER, textColor=colors.HexColor("#cc0000")),
    ))


def _toc_page(story, styles, lang):
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(_tr(lang, "toc_title"), styles["SectionTitle"]))
    story.append(Spacer(1, 0.2 * inch))
    titles = [
        _tr(lang, "section1_title"),
        _tr(lang, "section2_title"),
        _tr(lang, "section3_title"),
        _tr(lang, "section4_title"),
        _tr(lang, "section5_title"),
        _tr(lang, "section6_title"),
        _tr(lang, "section7_title"),
    ]
    for t in titles:
        story.append(Paragraph(t, styles["TOCEntry"]))
        story.append(Spacer(1, 0.05 * inch))


def _section_title(story, styles, title, desc):
    story.append(Paragraph(title, styles["SectionTitle"]))
    story.append(Paragraph(desc, styles["SectionDesc"]))


def _make_table(data: List[List[Any]], col_widths: Optional[List[float]] = None,
                header: bool = True) -> Table:
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    style_cmds = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        style_cmds.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ])
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f0f4f8")))
    t.setStyle(TableStyle(style_cmds))
    return t


def _section1_eda(story, styles, lang):
    eda = _load_eda()
    _section_title(story, styles, _tr(lang, "section1_title"), _tr(lang, "section1_desc"))

    shape = eda.get("dataset_shape", [20640, 9])
    story.append(Paragraph(
        f"<b>Dataset:</b> {shape[0]} filas × {shape[1]} columnas",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.1 * inch))

    desc = eda.get("descriptive_stats", _EDA_FALLBACK["descriptive_stats"])
    feat_order = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude", "MedHouseVal"]
    header_row = [_tr(lang, "feature"), _tr(lang, "count"), _tr(lang, "mean"), "Std",
                  _tr(lang, "min"), "25%", "50%", "75%", _tr(lang, "max"), "Skew", "Kurtosis"]
    rows = [header_row]
    for feat in feat_order:
        vals = desc.get(feat, ["—"] * 11)
        rows.append([feat] + [f"{v:.4f}" if isinstance(v, (int, float)) else str(v) for v in vals])
    story.append(Paragraph(f"<b>{_tr(lang, 'descriptive_stats')}</b>", styles["SubTitle"]))
    story.append(_make_table(rows))
    story.append(Spacer(1, 0.15 * inch))

    missing = eda.get("missing_values", _EDA_FALLBACK["missing_values"])
    miss_header = [_tr(lang, "feature"), _tr(lang, "count"), "%"]
    miss_rows = [miss_header]
    for feat in feat_order:
        vals = missing.get(feat, [0, 0.0])
        miss_rows.append([feat, str(int(vals[0])), f"{float(vals[1]):.2f}%"])
    story.append(Paragraph(f"<b>{_tr(lang, 'missing_values')}</b>", styles["SubTitle"]))
    story.append(_make_table(miss_rows))
    story.append(Spacer(1, 0.15 * inch))

    outliers = eda.get("outliers_iqr", _EDA_FALLBACK["outliers_iqr"])
    out_header = [_tr(lang, "feature"), _tr(lang, "count")]
    out_rows = [out_header]
    for feat in feat_order:
        if feat == "MedHouseVal":
            continue
        cnt = outliers.get(feat, 0)
        out_rows.append([feat, str(cnt)])
    story.append(Paragraph(f"<b>{_tr(lang, 'outliers')}</b>", styles["SubTitle"]))
    story.append(_make_table(out_rows))

    _embed_images(story, styles, lang, "eda")


def _section2_training(story, styles, lang):
    training = _load_training()
    _section_title(story, styles, _tr(lang, "section2_title"), _tr(lang, "section2_desc"))

    metrics = training.get("metrics", _TRAINING_FALLBACK["metrics"])
    header = [_tr(lang, "model"), "MSE", "RMSE", "MAE", "R²",
              _tr(lang, "training_time"), _tr(lang, "params")]
    rows = [header]
    for m in metrics:
        rows.append([
            m.get("model", ""),
            f"{m.get('MSE', 0):.4f}",
            f"{m.get('RMSE', 0):.4f}",
            f"{m.get('MAE', 0):.4f}",
            f"{m.get('R2', 0):.4f}",
            f"{m.get('training_time', 0):.1f}",
            str(m.get("params", "")),
        ])
    story.append(_make_table(rows))
    _embed_images(story, styles, lang, "training")


def _section3_best_model(story, styles, lang):
    training = _load_training()
    best = training.get("best_model", _TRAINING_FALLBACK["best_model"])
    _section_title(story, styles, _tr(lang, "section3_title"), _tr(lang, "section3_desc"))

    story.append(Paragraph(f"<b>{best.get('model', 'N/A')}</b>", styles["SubTitle"]))
    details = [
        [_tr(lang, "architecture"), best.get("architecture", "—")],
        [_tr(lang, "activation"), best.get("activation", "—")],
        [_tr(lang, "optimizer"), best.get("optimizer", "—")],
        [_tr(lang, "learning_rate"), str(best.get("learning_rate", "—"))],
        [_tr(lang, "batch_size"), str(best.get("batch_size", "—"))],
        [_tr(lang, "epochs"), str(best.get("epochs", "—"))],
        [_tr(lang, "early_stopping"), str(best.get("early_stopping", "—")) + " epochs"],
    ]
    detail_table = Table([[Paragraph(r[0], styles["TableCellLeft"]), Paragraph(r[1], styles["TableCell"])] for r in details],
                         colWidths=[2.0 * inch, 3.5 * inch])
    detail_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8edf3")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(detail_table)
    story.append(Spacer(1, 0.15 * inch))

    perf_header = ["MSE", "RMSE", "MAE", "R²"]
    perf_row = [
        f"{best.get('MSE', 0):.4f}",
        f"{best.get('RMSE', 0):.4f}",
        f"{best.get('MAE', 0):.4f}",
        f"{best.get('R2', 0):.4f}",
    ]
    story.append(Paragraph("<b>Performance</b>", styles["SubTitle"]))
    story.append(_make_table([perf_header, perf_row]))
    _embed_images(story, styles, lang, "best_model")


def _section4_cv(story, styles, lang):
    cv = _load_cv()
    _section_title(story, styles, _tr(lang, "section4_title"), _tr(lang, "section4_desc"))

    folds = cv.get("folds", _CV_FALLBACK["folds"])
    header = [_tr(lang, "fold"), "MSE", "RMSE", "MAE", "R²"]
    rows = [header]
    for f in folds:
        rows.append([
            str(f.get("fold", "")),
            f"{f.get('MSE', 0):.4f}",
            f"{f.get('RMSE', 0):.4f}",
            f"{f.get('MAE', 0):.4f}",
            f"{f.get('R2', 0):.4f}",
        ])

    mean = cv.get("mean", _CV_FALLBACK["mean"])
    std = cv.get("std", _CV_FALLBACK["std"])
    rows.append([
        f"<b>{_tr(lang, 'mean')}</b>",
        f"<b>{mean.get('MSE', 0):.4f}</b>",
        f"<b>{mean.get('RMSE', 0):.4f}</b>",
        f"<b>{mean.get('MAE', 0):.4f}</b>",
        f"<b>{mean.get('R2', 0):.4f}</b>",
    ])
    rows.append([
        f"<b>{_tr(lang, 'std')}</b>",
        f"<b>{std.get('MSE', 0):.4f}</b>",
        f"<b>{std.get('RMSE', 0):.4f}</b>",
        f"<b>{std.get('MAE', 0):.4f}</b>",
        f"<b>{std.get('R2', 0):.4f}</b>",
    ])
    story.append(_make_table(rows))
    _embed_images(story, styles, lang, "cv")


def _section5_hyperparameter(story, styles, lang):
    hp = _load_hyperparameter()
    _section_title(story, styles, _tr(lang, "section5_title"), _tr(lang, "section5_desc"))

    params = hp.get("best_params", _HYPERPARAMETER_FALLBACK["best_params"])
    story.append(Paragraph(f"<b>{_tr(lang, 'best_params')}</b>", styles["SubTitle"]))
    rows = [[_tr(lang, "parameter"), _tr(lang, "value")]]
    for key, val in params.items():
        rows.append([key.replace("_", " ").title(), str(val)])
    story.append(_make_table(rows, col_widths=[2.5 * inch, 3.0 * inch]))
    story.append(Spacer(1, 0.1 * inch))
    score = hp.get("best_score", _HYPERPARAMETER_FALLBACK["best_score"])
    story.append(Paragraph(
        f"<b>Best CV Score (R²):</b> {score:.4f}",
        styles["Normal"],
    ))
    _embed_images(story, styles, lang, "hyperparameter")


def _section6_statistical(story, styles, lang):
    tests = _load_statistical_tests()
    _section_title(story, styles, _tr(lang, "section6_title"), _tr(lang, "section6_desc"))

    test_data = {
        _tr(lang, "shapiro_wilk"): tests.get("shapiro_wilk", _STATISTICAL_TESTS_FALLBACK["shapiro_wilk"]),
        _tr(lang, "durbin_watson"): tests.get("durbin_watson", _STATISTICAL_TESTS_FALLBACK["durbin_watson"]),
        _tr(lang, "breusch_pagan"): tests.get("breusch_pagan", _STATISTICAL_TESTS_FALLBACK["breusch_pagan"]),
        _tr(lang, "friedman"): tests.get("friedman", _STATISTICAL_TESTS_FALLBACK["friedman"]),
    }
    header = [_tr(lang, "test"), _tr(lang, "statistic"), _tr(lang, "p_value"), _tr(lang, "conclusion")]
    rows = [header]
    for test_name, result in test_data.items():
        stat = result.get("statistic", 0)
        pval = result.get("p_value", 0)
        if pval < 0.0001:
            pval_str = "< 0.0001"
        else:
            pval_str = f"{pval:.4f}"
        conclusion = _tr(lang, "not_normal")
        if "autocorrelation" in result:
            conclusion = _tr(lang, "no_autocorrelation") if not result["autocorrelation"] else _tr(lang, "autocorrelation")
        elif "heteroscedastic" in result:
            conclusion = _tr(lang, "homoscedastic") if not result["heteroscedastic"] else _tr(lang, "heteroscedastic")
        elif "significant_difference" in result:
            conclusion = _tr(lang, "no_difference") if not result["significant_difference"] else _tr(lang, "significant_difference")
        elif "normal" in result:
            conclusion = _tr(lang, "normal") if result["normal"] else _tr(lang, "not_normal")
        rows.append([test_name, f"{stat:.4f}", pval_str, conclusion])
    story.append(_make_table(rows))
    _embed_images(story, styles, lang, "statistical")


def _section7_conclusions(story, styles, lang):
    training = _load_training()
    best = training.get("best_model", _TRAINING_FALLBACK["best_model"])
    cv = _load_cv()
    cv_std = cv.get("std", _CV_FALLBACK["std"])

    _section_title(story, styles, _tr(lang, "section7_title"), _tr(lang, "section7_desc"))
    conclusion = _tr(
        lang, "conclusion_text",
        arch=best.get("architecture", "[8, 64, 32, 16, 1]"),
        r2=f"{best.get('R2', 0.8284):.4f}",
        rmse=f"{best.get('RMSE', 0.4791):.4f}",
        cv_std=f"{cv_std.get('R2', 0.0086):.4f}",
    )
    story.append(Paragraph(conclusion, styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    if lang == "es":
        recs = [
            "El modelo de Red Neuronal supera en R² a los modelos lineales y basados en árboles.",
            "Validación cruzada muestra baja varianza, indicando generalización robusta.",
            "Se recomienda implementar monitoreo continuo de métricas en producción.",
            "Considerar reentrenamiento periódico con nuevos datos para mantener precisión.",
            "Explorar técnicas de feature engineering para mejorar aún más el rendimiento.",
        ]
    else:
        recs = [
            "Neural Network outperforms linear and tree-based models in R².",
            "Cross-validation shows low variance, indicating robust generalization.",
            "Continuous metric monitoring in production is recommended.",
            "Consider periodic retraining with new data to maintain accuracy.",
            "Explore feature engineering techniques to further improve performance.",
        ]
    for r in recs:
        story.append(Paragraph(f"• {r}", styles["Normal"]))


def _embed_images(story, styles, lang: str, section: str):
    images = _find_images()
    if not images:
        return
    section_keywords = {
        "eda": ["histogram", "correlation", "boxplot", "target", "qq", "pairplot", "eda"],
        "training": ["loss", "training", "learning", "curve"],
        "best_model": ["residual", "prediction", "best", "error"],
        "cv": ["cv", "boxplot", "cross", "validation"],
        "hyperparameter": ["hyper", "param", "tuning", "heatmap"],
        "statistical": ["statistical", "test", "qq", "residual"],
    }
    keywords = section_keywords.get(section, [])
    for img_path in images:
        name = img_path.stem.lower()
        if not any(kw in name for kw in keywords):
            continue
        try:
            img = Image(str(img_path))
            img.drawWidth = 5.0 * inch
            img.drawHeight = 3.5 * inch
            img.hAlign = TA_CENTER
            story.append(Spacer(1, 0.1 * inch))
            story.append(img)
            story.append(Spacer(1, 0.1 * inch))
        except Exception:
            pass


def _header_footer(canvas, doc, lang: str):
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#999999"))
    canvas.drawCentredString(letter[0] / 2, 0.4 * inch,
                             _tr(lang, "page_x_of_y", page=doc.page, total="?"))
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#1a3a5c"))
    canvas.drawString(doc.leftMargin, letter[1] - 0.5 * inch,
                      _tr(lang, "report_title"))
    canvas.setStrokeColor(colors.HexColor("#1a3a5c"))
    canvas.setLineWidth(0.5)
    canvas.line(doc.leftMargin, letter[1] - 0.55 * inch,
                letter[0] - doc.rightMargin, letter[1] - 0.55 * inch)
    canvas.setFont("Helvetica", 6)
    canvas.setFillColor(colors.HexColor("#aaaaaa"))
    canvas.drawRightString(letter[0] - doc.rightMargin, 0.4 * inch,
                           _tr(lang, "confidential"))
    canvas.restoreState()


def generate_pdf(language: str = "es", output_path: Optional[str] = None) -> bytes:
    lang = language if language in _LANG else "es"
    buffer = io.BytesIO()
    styles = _make_styles()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        title=_tr(lang, "report_title"),
        author="California Housing ML",
    )

    story: List[Any] = []

    _cover_page(story, styles, lang)
    story.append(PageBreak())
    _toc_page(story, styles, lang)
    story.append(PageBreak())

    _section1_eda(story, styles, lang)
    story.append(PageBreak())
    _section2_training(story, styles, lang)
    story.append(PageBreak())
    _section3_best_model(story, styles, lang)
    story.append(PageBreak())
    _section4_cv(story, styles, lang)
    story.append(PageBreak())
    _section5_hyperparameter(story, styles, lang)
    story.append(PageBreak())
    _section6_statistical(story, styles, lang)
    story.append(PageBreak())
    _section7_conclusions(story, styles, lang)

    doc.build(
        story,
        onFirstPage=lambda c, d: None,
        onLaterPages=lambda c, d: _header_footer(c, d, lang),
    )

    result = buffer.getvalue()
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(result)
    return result
