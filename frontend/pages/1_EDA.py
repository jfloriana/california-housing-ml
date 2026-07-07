import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils import api_client

if not st.session_state.get("authenticated"):
    st.warning("Please login")
    st.stop()

LANG = {
    "es": {
        "title": "Análisis Exploratorio de Datos",
        "subtitle": "Exploración completa del dataset California Housing",
        "dataset_info": "Información del Dataset",
        "shape": "Dimensiones",
        "rows": "Filas",
        "cols": "Columnas",
        "columns": "Columnas",
        "dtypes": "Tipos de Datos",
        "descriptive_stats": "Estadísticas Descriptivas",
        "missing_duplicates": "Valores Faltantes y Duplicados",
        "missing": "Valores Faltantes",
        "duplicates": "Filas Duplicadas",
        "outliers": "Detección de Valores Atípicos (Outliers)",
        "outliers_iqr": "Outliers (IQR)",
        "outliers_zscore": "Outliers (Z-Score)",
        "correlation_matrix": "Matriz de Correlación",
        "correlation_target": "Correlación con la Variable Objetivo",
        "interpretation": "Interpretación de Resultados",
        "no_data": "No se pudieron cargar los datos de EDA",
        "loading_error": "Error al cargar datos de EDA",
        "no_outlier_data": "No hay datos de outliers disponibles",
        "no_corr_data": "No hay datos de correlación disponibles",
        "total": "Total",
        "feature": "Característica",
        "correlation": "Correlación",
        "variable": "Variable",
        "type": "Tipo",
        "count": "Conteo",
        "mean": "Media",
        "std": "Desviación Estándar",
        "min": "Mínimo",
        "max": "Máximo",
        "p25": "25%",
        "p50": "50% (Mediana)",
        "p75": "75%",
        "skew": "Asimetría (Skewness)",
        "kurt": "Curtosis",
        "method": "Método",
        "value": "Valor",
        "interpretation_text": "El dataset de California Housing contiene {rows} registros y {cols} características. Se detectaron {outliers_iqr} outliers por el método IQR y {outliers_zscore} por Z-Score. Las correlaciones más fuertes con el valor de la vivienda son: {top_corr}.",
    },
    "en": {
        "title": "Exploratory Data Analysis",
        "subtitle": "Complete exploration of the California Housing dataset",
        "dataset_info": "Dataset Information",
        "shape": "Shape",
        "rows": "Rows",
        "cols": "Columns",
        "columns": "Columns",
        "dtypes": "Data Types",
        "descriptive_stats": "Descriptive Statistics",
        "missing_duplicates": "Missing Values and Duplicates",
        "missing": "Missing Values",
        "duplicates": "Duplicate Rows",
        "outliers": "Outlier Detection",
        "outliers_iqr": "Outliers (IQR)",
        "outliers_zscore": "Outliers (Z-Score)",
        "correlation_matrix": "Correlation Matrix",
        "correlation_target": "Correlation with Target Variable",
        "interpretation": "Results Interpretation",
        "no_data": "Could not load EDA data",
        "loading_error": "Error loading EDA data",
        "no_outlier_data": "No outlier data available",
        "no_corr_data": "No correlation data available",
        "total": "Total",
        "feature": "Feature",
        "correlation": "Correlation",
        "variable": "Variable",
        "type": "Type",
        "count": "Count",
        "mean": "Mean",
        "std": "Standard Deviation",
        "min": "Minimum",
        "max": "Maximum",
        "p25": "25%",
        "p50": "50% (Median)",
        "p75": "75%",
        "skew": "Skewness",
        "kurt": "Kurtosis",
        "method": "Method",
        "value": "Value",
        "interpretation_text": "The California Housing dataset contains {rows} records and {cols} features. {outliers_iqr} outliers were detected by the IQR method and {outliers_zscore} by the Z-Score method. The strongest correlations with housing value are: {top_corr}.",
    },
}

def tr(key):
    return LANG.get(st.session_state.language, LANG["es"]).get(key, key)

lang = st.session_state.language
st.title(tr("title"))
st.markdown(f"*{tr('subtitle')}*")
st.markdown("---")

try:
    with st.spinner("Loading EDA data..."):
        data = api_client.get_eda_results()
except Exception as e:
    st.error(f"{tr('loading_error')}: {e}")
    st.stop()

# ── Section 1: Dataset Info ─────────────────────────────────────
st.header(tr("dataset_info"))

col1, col2, col3 = st.columns(3)
shape = data.get("shape", {})
with col1:
    st.metric(tr("rows"), shape.get("rows", "N/A"))
with col2:
    st.metric(tr("cols"), shape.get("columns", "N/A"))
with col3:
    st.metric(tr("total") + " " + tr("columns"), shape.get("columns", "N/A"))

col_info = data.get("columns_info", data.get("dtypes", {}))
if isinstance(col_info, dict):
    df_cols = pd.DataFrame([
        {"Variable": k, tr("type"): v} for k, v in col_info.items()
    ])
    st.dataframe(df_cols, use_container_width=True)

# ── Section 2: Descriptive Statistics ───────────────────────────
st.header(tr("descriptive_stats"))
desc = data.get("descriptive_stats", data.get("statistics", {}))
if isinstance(desc, dict):
    df_desc = pd.DataFrame(desc).T if not isinstance(desc, list) else pd.DataFrame(desc)
    st.dataframe(df_desc.style.format("{:.4f}"), use_container_width=True)
elif isinstance(desc, list):
    st.dataframe(pd.DataFrame(desc).style.format("{:.4f}"), use_container_width=True)

# ── Section 3: Missing Values and Duplicates ────────────────────
st.header(tr("missing_duplicates"))
missing = data.get("missing_values", data.get("missing", {}))
duplicates = data.get("duplicates", data.get("duplicated", {}))

c1, c2 = st.columns(2)
with c1:
    if isinstance(missing, dict):
        df_miss = pd.DataFrame([{tr("variable"): k, tr("count"): v} for k, v in missing.items()])
        st.dataframe(df_miss, use_container_width=True)
    elif isinstance(missing, (int, float)):
        st.metric(tr("missing"), missing)
    else:
        st.metric(tr("missing"), str(missing))

with c2:
    if isinstance(duplicates, dict):
        df_dup = pd.DataFrame([{tr("variable"): k, tr("count"): v} for k, v in duplicates.items()])
        st.dataframe(df_dup, use_container_width=True)
    elif isinstance(duplicates, (int, float)):
        st.metric(tr("duplicates"), duplicates)
    else:
        st.metric(tr("duplicates"), str(duplicates))

# ── Section 4: Outliers ─────────────────────────────────────────
st.header(tr("outliers"))
outliers_data = data.get("outliers", data.get("outlier_detection", {}))
if outliers_data:
    iqr = outliers_data.get("iqr", outliers_data.get("IQR", {}))
    zscore = outliers_data.get("zscore", outliers_data.get("z_score", {}))

    if isinstance(iqr, dict) and isinstance(zscore, dict):
        all_vars = sorted(set(list(iqr.keys()) + list(zscore.keys())))
        df_out = pd.DataFrame({
            tr("variable"): all_vars,
            tr("outliers_iqr"): [iqr.get(v, 0) for v in all_vars],
            tr("outliers_zscore"): [zscore.get(v, 0) for v in all_vars],
        })
        fig_out = px.bar(
            df_out,
            x=tr("variable"),
            y=[tr("outliers_iqr"), tr("outliers_zscore")],
            barmode="group",
            title=tr("outliers"),
        )
        st.plotly_chart(fig_out, use_container_width=True)
    elif isinstance(iqr, (int, float)) or isinstance(zscore, (int, float)):
        df_out = pd.DataFrame({
            tr("method"): [tr("outliers_iqr"), tr("outliers_zscore")],
            tr("value"): [iqr if isinstance(iqr, (int, float)) else 0, zscore if isinstance(zscore, (int, float)) else 0],
        })
        fig_out = px.bar(df_out, x=tr("method"), y=tr("value"), title=tr("outliers"), color=tr("method"))
        st.plotly_chart(fig_out, use_container_width=True)
    else:
        st.info(tr("no_outlier_data"))
else:
    st.info(tr("no_outlier_data"))

# ── Section 5: Correlation Matrix ───────────────────────────────
st.header(tr("correlation_matrix"))
corr_data = data.get("correlation_matrix", data.get("correlation", data.get("corr_matrix", {})))
if corr_data:
    if isinstance(corr_data, dict):
        df_corr = pd.DataFrame(corr_data).astype(float)
    elif isinstance(corr_data, list):
        df_corr = pd.DataFrame(corr_data).astype(float)
    else:
        df_corr = None

    if df_corr is not None and not df_corr.empty:
        fig_heat = go.Figure(data=go.Heatmap(
            z=df_corr.values,
            x=df_corr.columns.tolist(),
            y=df_corr.index.tolist(),
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=np.round(df_corr.values, 2),
            texttemplate="%{text}",
        ))
        fig_heat.update_layout(title=tr("correlation_matrix"), height=600)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info(tr("no_corr_data"))
else:
    st.info(tr("no_corr_data"))

# ── Section 6: Correlation with Target ──────────────────────────
st.header(tr("correlation_target"))
target_corr = data.get("correlation_with_target", data.get("target_correlation", data.get("corr_with_target", {})))
if not target_corr and df_corr is not None and not df_corr.empty:
    target_col = "MedHouseVal" if "MedHouseVal" in df_corr.columns else df_corr.columns[-1]
    if target_col in df_corr.columns:
        target_corr = df_corr[target_col].drop(target_col).to_dict()

if target_corr:
    if isinstance(target_corr, dict):
        df_tc = pd.DataFrame({
            tr("feature"): list(target_corr.keys()),
            tr("correlation"): list(target_corr.values()),
        }).sort_values(tr("correlation"), key=abs, ascending=True)
        color_col = tr("correlation")
        fig_tc = px.bar(
            df_tc,
            x=tr("correlation"),
            y=tr("feature"),
            orientation="h",
            title=tr("correlation_target"),
            color=color_col,
            color_continuous_scale="RdBu_r",
            range_color=[-1, 1],
        )
        fig_tc.update_layout(height=400)
        st.plotly_chart(fig_tc, use_container_width=True)
    elif isinstance(target_corr, list):
        df_tc = pd.DataFrame(target_corr)
        if not df_tc.empty:
            st.dataframe(df_tc.style.format("{:.4f}"), use_container_width=True)

# ── Section 7: Interpretation ───────────────────────────────────
st.header(tr("interpretation"))
rows_val = shape.get("rows", "N/A")
cols_val = shape.get("columns", "N/A")
top_corr_text = tr("no_corr_data")
if target_corr:
    if isinstance(target_corr, dict):
        sorted_corr = sorted(target_corr.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        top_corr_text = ", ".join([f"{k}: {v:.3f}" for k, v in sorted_corr])
    elif isinstance(target_corr, list) and len(target_corr) > 0:
        top_corr_text = str(target_corr[:3])

iqr_total = 0
zscore_total = 0
if outliers_data:
    iq = outliers_data.get("iqr", outliers_data.get("IQR", {}))
    zs = outliers_data.get("zscore", outliers_data.get("z_score", {}))
    if isinstance(iq, dict):
        iqr_total = sum(v for v in iq.values() if isinstance(v, (int, float)))
    elif isinstance(iq, (int, float)):
        iqr_total = iq
    if isinstance(zs, dict):
        zscore_total = sum(v for v in zs.values() if isinstance(v, (int, float)))
    elif isinstance(zs, (int, float)):
        zscore_total = zs

st.info(
    tr("interpretation_text").format(
        rows=rows_val, cols=cols_val,
        outliers_iqr=iqr_total, outliers_zscore=zscore_total,
        top_corr=top_corr_text,
    )
)
