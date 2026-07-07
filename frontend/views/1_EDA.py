import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

if not st.session_state.get("authenticated"):
    st.warning("Please login")
    st.stop()

LANG = {
    "es": {
        "title": "Análisis Exploratorio de Datos Financieros",
        "subtitle": "Sube un dataset CSV financiero para analizarlo",
        "upload_header": "Cargar Dataset Financiero",
        "upload_hint": "Sube un archivo CSV con datos numéricos para pronóstico financiero",
        "uploaded_file": "Archivo cargado",
        "target_select": "Selecciona la variable a pronosticar (target)",
        "start_eda": "Iniciar EDA",
        "dataset_info": "Información del Dataset",
        "rows": "Filas",
        "cols": "Columnas",
        "columns": "Columnas",
        "descriptive_stats": "Estadísticas Descriptivas",
        "missing_duplicates": "Valores Faltantes y Duplicados",
        "missing": "Valores Faltantes",
        "duplicates": "Filas Duplicadas",
        "outliers": "Valores Atípicos (Outliers por IQR)",
        "correlation_matrix": "Matriz de Correlación",
        "correlation_target": "Correlación con Variable a Pronosticar",
        "interpretation": "Interpretación",
        "no_data": "No hay datos cargados. Sube un CSV financiero para comenzar.",
        "loading_error": "Error al procesar el archivo",
        "feature": "Variable",
        "correlation": "Correlación",
        "variable": "Variable",
        "count": "Conteo",
        "mean": "Media",
        "std": "Desviación Estándar",
        "min": "Mínimo",
        "max": "Máximo",
        "loading_data": "Procesando datos...",
        "target": "Objetivo",
        "samples": "Muestras",
        "numeric_cols": "Columnas Numéricas",
        "interpretation_text": "El dataset financiero contiene {samples} registros y {features} variables. La variable a pronosticar es '{target}'.",
    },
    "en": {
        "title": "Financial Data Exploratory Analysis",
        "subtitle": "Upload a financial CSV dataset for analysis",
        "upload_header": "Upload Financial Dataset",
        "upload_hint": "Upload a CSV file with numeric data for financial forecasting",
        "uploaded_file": "Uploaded file",
        "target_select": "Select the variable to forecast (target)",
        "start_eda": "Run EDA",
        "dataset_info": "Dataset Information",
        "rows": "Rows",
        "cols": "Columns",
        "columns": "Columns",
        "descriptive_stats": "Descriptive Statistics",
        "missing_duplicates": "Missing Values and Duplicates",
        "missing": "Missing Values",
        "duplicates": "Duplicate Rows",
        "outliers": "Outliers (IQR)",
        "correlation_matrix": "Correlation Matrix",
        "correlation_target": "Correlation with Forecast Target",
        "interpretation": "Interpretation",
        "no_data": "No data loaded. Upload a financial CSV to begin.",
        "loading_error": "Error processing file",
        "feature": "Feature",
        "correlation": "Correlation",
        "variable": "Variable",
        "count": "Count",
        "mean": "Mean",
        "std": "Standard Deviation",
        "min": "Minimum",
        "max": "Maximum",
        "loading_data": "Processing data...",
        "target": "Target",
        "samples": "Samples",
        "numeric_cols": "Numeric Columns",
        "interpretation_text": "The financial dataset contains {samples} records and {features} variables. The forecast target is '{target}'.",
    },
}

def tr(key):
    return LANG.get(st.session_state.language, LANG["es"]).get(key, key)

lang = st.session_state.language
st.title(tr("title"))
st.markdown(f"*{tr('subtitle')}*")
st.markdown("---")

# ── Upload section ──
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "target_col" not in st.session_state:
    st.session_state.target_col = None

uploaded_file = st.file_uploader(tr("upload_header"), type="csv", help=tr("upload_hint"))

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            st.error("El dataset debe tener al menos 2 columnas numéricas.")
            st.stop()
        st.success(f"{tr('uploaded_file')}: {uploaded_file.name} ({df.shape[0]} rows, {df.shape[1]} cols)")

        target = st.selectbox(tr("target_select"), numeric_cols, index=len(numeric_cols)-1)

        if st.button(tr("start_eda"), type="primary", use_container_width=True):
            st.session_state.uploaded_df = df
            st.session_state.target_col = target
            st.rerun()
    except Exception as e:
        st.error(f"{tr('loading_error')}: {e}")
        st.stop()

df = st.session_state.get("uploaded_df")
target_col = st.session_state.get("target_col")

if df is None:
    st.info(tr("no_data"))
    st.stop()

# Ensure only numeric columns
df_num = df.select_dtypes(include=[np.number])
if target_col not in df_num.columns:
    st.error(f"Target '{target_col}' no es numérica.")
    st.stop()

X = df_num.drop(columns=[target_col])
y = df_num[target_col]

# Store for other pages
st.session_state.X_data = X
st.session_state.y_data = y
st.session_state.feature_names = X.columns.tolist()

st.markdown("---")

# ── 1. Dataset Info ──
st.header(tr("dataset_info"))
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(tr("samples"), f"{df.shape[0]:,}")
with col2:
    st.metric(tr("numeric_cols"), len(X.columns))
with col3:
    st.metric(tr("target"), target_col)

st.dataframe(pd.DataFrame({
    tr("variable"): df_num.columns,
    tr("count"): [dt.name for dt in df_num.dtypes],
}).set_index(tr("variable")), use_container_width=True)

# ── 2. Descriptive Stats ──
st.header(tr("descriptive_stats"))
desc = df_num.describe().T
desc["skew"] = df_num.skew()
desc["kurtosis"] = df_num.kurtosis()
st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)

# ── 3. Missing & Duplicates ──
st.header(tr("missing_duplicates"))
c1, c2 = st.columns(2)
with c1:
    missing_s = df.isnull().sum()
    if missing_s.sum() > 0:
        st.dataframe(pd.DataFrame({tr("variable"): missing_s.index, tr("count"): missing_s.values}), use_container_width=True)
    else:
        st.metric(tr("missing"), 0)
with c2:
    dup_count = df.duplicated().sum()
    st.metric(tr("duplicates"), dup_count)

# ── 4. Outliers ──
st.header(tr("outliers"))
outlier_counts = {}
for col in X.columns:
    Q1 = df_num[col].quantile(0.25)
    Q3 = df_num[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df_num[col] < (Q1 - 1.5*IQR)) | (df_num[col] > (Q3 + 1.5*IQR))).sum()
    outlier_counts[col] = int(outliers)

df_out = pd.DataFrame({tr("feature"): list(outlier_counts.keys()), tr("count"): list(outlier_counts.values())})
fig_out = px.bar(df_out, x=tr("feature"), y=tr("count"), title=tr("outliers"), color=tr("count"),
                 color_continuous_scale="reds")
st.plotly_chart(fig_out, use_container_width=True)

# ── 5. Correlation Matrix ──
st.header(tr("correlation_matrix"))
corr_matrix = df_num.corr()
fig_heat = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns.tolist(),
    y=corr_matrix.index.tolist(),
    colorscale="RdBu_r", zmin=-1, zmax=1,
    text=np.round(corr_matrix.values, 2), texttemplate="%{text}",
))
fig_heat.update_layout(height=600, title=tr("correlation_matrix"))
st.plotly_chart(fig_heat, use_container_width=True)

# ── 6. Correlation with Target ──
st.header(tr("correlation_target"))
target_corr = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=True)
df_tc = pd.DataFrame({tr("feature"): target_corr.index, tr("correlation"): target_corr.values})
fig_tc = px.bar(df_tc, x=tr("correlation"), y=tr("feature"), orientation="h",
                title=tr("correlation_target"), color=tr("correlation"),
                color_continuous_scale="RdBu_r", range_color=[-1, 1])
fig_tc.update_layout(height=400)
st.plotly_chart(fig_tc, use_container_width=True)

# ── 7. Interpretation ──
st.header(tr("interpretation"))
st.info(tr("interpretation_text").format(
    samples=f"{df.shape[0]:,}",
    features=len(X.columns),
    target=target_col,
))
