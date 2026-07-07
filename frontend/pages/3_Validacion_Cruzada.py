import streamlit as st
import pandas as pd
import plotly.express as px
from utils import api_client

if not st.session_state.get("authenticated"):
    st.warning("Please login")
    st.stop()

LANG = {
    "es": {
        "title": "Validación Cruzada",
        "subtitle": "Evaluación mediante K-Fold Cross Validation",
        "config": "Configuración",
        "folds": "Número de Folds",
        "scoring": "Métrica de Evaluación",
        "shuffle": "Shuffle",
        "random_state": "Random State",
        "fold_results": "Resultados por Fold",
        "fold": "Fold",
        "mse": "MSE",
        "r2": "R²",
        "boxplot": "Distribución de Resultados",
        "summary": "Resumen Estadístico",
        "mean": "Media",
        "std": "Desviación Estándar",
        "interpretation": "Interpretación de Resultados",
        "no_data": "No se pudieron cargar los datos de validación cruzada",
        "loading_error": "Error al cargar datos de validación cruzada",
        "interpretation_text": "La validación cruzada con {folds} folds muestra un MSE promedio de {mean_mse:.4f} (±{std_mse:.4f}) y un R² promedio de {mean_r2:.4f} (±{std_r2:.4f}). La consistencia entre folds indica que el modelo es {stable}.",
        "stable": "estable y generaliza bien",
        "unstable": "potencialmente inestable",
    },
    "en": {
        "title": "Cross Validation",
        "subtitle": "K-Fold Cross Validation evaluation",
        "config": "Configuration",
        "folds": "Number of Folds",
        "scoring": "Scoring Metric",
        "shuffle": "Shuffle",
        "random_state": "Random State",
        "fold_results": "Per-Fold Results",
        "fold": "Fold",
        "mse": "MSE",
        "r2": "R²",
        "boxplot": "Result Distribution",
        "summary": "Statistical Summary",
        "mean": "Mean",
        "std": "Standard Deviation",
        "interpretation": "Results Interpretation",
        "no_data": "Could not load cross validation data",
        "loading_error": "Error loading cross validation data",
        "interpretation_text": "Cross validation with {folds} folds shows an average MSE of {mean_mse:.4f} (±{std_mse:.4f}) and an average R² of {mean_r2:.4f} (±{std_r2:.4f}). The consistency across folds indicates the model is {stable}.",
        "stable": "stable and generalizes well",
        "unstable": "potentially unstable",
    },
}

def tr(key):
    return LANG.get(st.session_state.language, LANG["es"]).get(key, key)

lang = st.session_state.language
st.title(tr("title"))
st.markdown(f"*{tr('subtitle')}*")
st.markdown("---")

try:
    with st.spinner("Loading cross validation data..."):
        data = api_client.get_cross_validation()
except Exception as e:
    st.error(f"{tr('loading_error')}: {e}")
    st.stop()

# ── Section 1: Configuration ────────────────────────────────────
st.header(tr("config"))
config = data.get("config", data.get("configuration", {}))
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric(tr("folds"), config.get("n_folds") or config.get("folds", "N/A"))
with c2:
    st.metric(tr("scoring"), config.get("scoring", "N/A"))
with c3:
    st.metric(tr("shuffle"), str(config.get("shuffle", "N/A")))
with c4:
    st.metric(tr("random_state"), config.get("random_state", "N/A"))

# ── Section 2: Per-Fold Results ─────────────────────────────────
st.header(tr("fold_results"))
folds = data.get("folds", data.get("results", data.get("cv_results", [])))
if folds:
    df_folds = pd.DataFrame(folds)
    fold_mse = next((c for c in ["MSE", "mse", "test_mse", "test_MSE"] if c in df_folds.columns), tr("mse"))
    fold_r2 = next((c for c in ["R2", "R²", "r2", "test_r2", "test_R2"] if c in df_folds.columns), tr("r2"))
    fold_label = next((c for c in ["fold", "Fold", "split", "Split"] if c in df_folds.columns), tr("fold"))

    col_map = {}
    if fold_label not in [tr("fold"), "fold", "Fold"]:
        pass

    display_cols = []
    if fold_label in df_folds.columns:
        display_cols.append(fold_label)
    if fold_mse in df_folds.columns:
        display_cols.append(fold_mse)
    if fold_r2 in df_folds.columns:
        display_cols.append(fold_r2)
    if not display_cols:
        display_cols = df_folds.columns.tolist()

    df_display = df_folds[display_cols].copy()
    rename_map = {}
    for old, new in [(fold_label, tr("fold")), (fold_mse, tr("mse")), (fold_r2, tr("r2"))]:
        if old in df_display.columns and old != new:
            rename_map[old] = new
    if rename_map:
        df_display = df_display.rename(columns=rename_map)

    format_cols = [c for c in [tr("mse"), tr("r2")] if c in df_display.columns]
    st.dataframe(df_display.style.format({tr("mse"): "{:.4f}", tr("r2"): "{:.4f}"}, subset=format_cols), use_container_width=True)

    # ── Section 3: Boxplot ──────────────────────────────────────────
    st.header(tr("boxplot"))
    fig_box = px.box(
        df_display.melt(id_vars=[tr("fold")], value_vars=[tr("mse"), tr("r2")]),
        x="variable", y="value", color="variable",
        title=tr("boxplot"),
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

    # ── Section 4: Summary ──────────────────────────────────────────
    st.header(tr("summary"))
    mean_mse = df_display[tr("mse")].mean() if tr("mse") in df_display.columns else 0
    std_mse = df_display[tr("mse")].std() if tr("mse") in df_display.columns else 0
    mean_r2 = df_display[tr("r2")].mean() if tr("r2") in df_display.columns else 0
    std_r2 = df_display[tr("r2")].std() if tr("r2") in df_display.columns else 0

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric(f"{tr('mean')} MSE", f"{mean_mse:.4f}")
    with sc2:
        st.metric(f"{tr('std')} MSE", f"{std_mse:.4f}")
    with sc3:
        st.metric(f"{tr('mean')} R²", f"{mean_r2:.4f}")
    with sc4:
        st.metric(f"{tr('std')} R²", f"{std_r2:.4f}")

else:
    summary = data.get("summary", data.get("mean", {}))
    if summary:
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.metric(f"{tr('mean')} MSE", f"{summary.get('mean_mse', summary.get('mse_mean', 'N/A'))}")
        with s2:
            st.metric(f"{tr('std')} MSE", f"{summary.get('std_mse', summary.get('mse_std', 'N/A'))}")
        with s3:
            st.metric(f"{tr('mean')} R²", f"{summary.get('mean_r2', summary.get('r2_mean', 'N/A'))}")
        with s4:
            st.metric(f"{tr('std')} R²", f"{summary.get('std_r2', summary.get('r2_std', 'N/A'))}")
    else:
        st.info(tr("no_data"))

# ── Section 5: Interpretation ──────────────────────────────────
st.header(tr("interpretation"))
n_folds = config.get("n_folds") or config.get("folds", "N/A")

if folds:
    df_folds = pd.DataFrame(folds)
    m_col = next((c for c in ["MSE", "mse", "test_mse"] if c in df_folds.columns), None)
    r_col = next((c for c in ["R2", "R²", "r2", "test_r2"] if c in df_folds.columns), None)
    mean_mse_v = df_folds[m_col].mean() if m_col else 0
    std_mse_v = df_folds[m_col].std() if m_col else 0
    mean_r2_v = df_folds[r_col].mean() if r_col else 0
    std_r2_v = df_folds[r_col].std() if r_col else 0
    stable_flag = tr("stable") if std_r2_v < 0.1 else tr("unstable")

    st.info(
        tr("interpretation_text").format(
            folds=n_folds,
            mean_mse=mean_mse_v,
            std_mse=std_mse_v,
            mean_r2=mean_r2_v,
            std_r2=std_r2_v,
            stable=stable_flag,
        )
    )
else:
    st.info(tr("no_data"))
