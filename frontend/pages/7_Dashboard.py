import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils import api_client

if not st.session_state.get("authenticated"):
    st.warning("Please login")
    st.stop()

LANG = {
    "es": {
        "title": "Dashboard General",
        "subtitle": "Resumen completo del proyecto California Housing ML",
        "total_models": "Total Modelos",
        "best_r2": "Mejor R²",
        "best_rmse": "Mejor RMSE",
        "dataset_size": "Tamaño Dataset",
        "model_comparison": "Comparación de Modelos",
        "top_correlations": "Principales Correlaciones",
        "stats_summary": "Resumen de Pruebas Estadísticas",
        "loading_error": "Error al cargar datos del dashboard",
        "no_data": "No hay datos disponibles",
        "shapiro": "Shapiro-Wilk",
        "dw": "Durbin-Watson",
        "bp": "Breusch-Pagan",
        "friedman": "Friedman",
        "normal": "Normal",
        "not_normal": "No normal",
        "no_auto": "Sin autocorr.",
        "homo": "Homoced.",
        "hetero": "Heteroced.",
        "sig_diff": "Dif. significativas",
        "no_sig": "Sin dif. significativas",
        "model": "Modelo",
        "r2": "R²",
        "rmse": "RMSE",
        "feature": "Variable",
        "correlation": "Correlación",
        "test": "Prueba",
        "result": "Resultado",
    },
    "en": {
        "title": "General Dashboard",
        "subtitle": "Complete summary of the California Housing ML project",
        "total_models": "Total Models",
        "best_r2": "Best R²",
        "best_rmse": "Best RMSE",
        "dataset_size": "Dataset Size",
        "model_comparison": "Model Comparison",
        "top_correlations": "Top Correlations",
        "stats_summary": "Statistical Tests Summary",
        "loading_error": "Error loading dashboard data",
        "no_data": "No data available",
        "shapiro": "Shapiro-Wilk",
        "dw": "Durbin-Watson",
        "bp": "Breusch-Pagan",
        "friedman": "Friedman",
        "normal": "Normal",
        "not_normal": "Not normal",
        "no_auto": "No autocorr.",
        "homo": "Homosced.",
        "hetero": "Heterosced.",
        "sig_diff": "Significant diff.",
        "no_sig": "No significant diff.",
        "model": "Model",
        "r2": "R²",
        "rmse": "RMSE",
        "feature": "Feature",
        "correlation": "Correlation",
        "test": "Test",
        "result": "Result",
    },
}

def tr(key):
    return LANG.get(st.session_state.language, LANG["es"]).get(key, key)

lang = st.session_state.language
st.title(tr("title"))
st.markdown(f"*{tr('subtitle')}*")
st.markdown("---")

try:
    with st.spinner("Loading dashboard data..."):
        all_data = api_client.get_all_metrics()
except Exception as e:
    st.error(f"{tr('loading_error')}: {e}")
    all_data = {}

# Extract data
eda = all_data.get("eda", all_data.get("eda_results", {}))
training_raw = all_data.get("training", all_data.get("training_metrics", []))
cv = all_data.get("cross_validation", all_data.get("cv", {}))
stats = all_data.get("statistical_tests", all_data.get("stats", {}))

# training_raw can be a list (from API) or dict with "metrics" key
if isinstance(training_raw, list):
    models_list = training_raw
elif isinstance(training_raw, dict):
    models_list = training_raw.get("metrics", training_raw.get("results", []))
else:
    models_list = []
n_models = len(models_list)

best_r2_val = "N/A"
best_rmse_val = "N/A"
if models_list:
    df_m = pd.DataFrame(models_list)
    r2_c = next((c for c in ["R2", "R²", "r2"] if c in df_m.columns), None)
    rmse_c = next((c for c in ["RMSE", "rmse"] if c in df_m.columns), None)
    if r2_c:
        best_r2_val = f"{df_m[r2_c].max():.4f}" if df_m[r2_c].dtype in ("float64", "int64") else str(df_m[r2_c].max())
    if rmse_c:
        best_rmse_val = f"{df_m[rmse_c].min():.4f}" if df_m[rmse_c].dtype in ("float64", "int64") else str(df_m[rmse_c].min())

shape = eda.get("shape", {})
dataset_size = shape.get("rows", "N/A")
if isinstance(dataset_size, int):
    dataset_size = f"{dataset_size:,}"

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""
        <div style="background:#1E3A5F;padding:20px;border-radius:10px;text-align:center;">
            <h3 style="color:white;margin:0;">{n_models}</h3>
            <p style="color:#A0C4FF;margin:0;">{tr('total_models')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div style="background:#1E5F3A;padding:20px;border-radius:10px;text-align:center;">
            <h3 style="color:white;margin:0;">{best_r2_val}</h3>
            <p style="color:#A0FFC4;margin:0;">{tr('best_r2')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
        <div style="background:#5F1E3A;padding:20px;border-radius:10px;text-align:center;">
            <h3 style="color:white;margin:0;">{best_rmse_val}</h3>
            <p style="color:#FFA0C4;margin:0;">{tr('best_rmse')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"""
        <div style="background:#3A5F1E;padding:20px;border-radius:10px;text-align:center;">
            <h3 style="color:white;margin:0;">{dataset_size}</h3>
            <p style="color:#C4FFA0;margin:0;">{tr('dataset_size')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Row 2: Model Comparison Bar Chart ──────────────────────────
st.subheader(f"📈 {tr('model_comparison')}")
if models_list:
    df_mc = pd.DataFrame(models_list)
    name_col = "name" if "name" in df_mc.columns else "model"
    r2c = next((c for c in ["R2", "R²", "r2"] if c in df_mc.columns), None)
    rmsec = next((c for c in ["RMSE", "rmse"] if c in df_mc.columns), None)

    if r2c and rmsec:
        df_plot = df_mc[[name_col, r2c, rmsec]].melt(
            id_vars=[name_col], var_name=tr("model"), value_name="Value"
        )
        fig_comp = px.bar(
            df_plot, x=name_col, y="Value", color=tr("model"),
            barmode="group", title=tr("model_comparison"),
            color_discrete_sequence=["#2E91E5", "#E76376"],
        )
        fig_comp.update_layout(xaxis_title=tr("model"), yaxis_title="Value")
        st.plotly_chart(fig_comp, use_container_width=True)
    else:
        st.dataframe(df_mc, use_container_width=True)
else:
    st.info(tr("no_data"))

# ── Row 3: Top Correlations ────────────────────────────────────
st.subheader(f"🔗 {tr('top_correlations')}")
corr_target = eda.get("correlation_with_target", eda.get("target_correlation", eda.get("corr_with_target", {})))
if not corr_target:
    corr_matrix = eda.get("correlation_matrix", eda.get("correlation", eda.get("corr_matrix", {})))
    if isinstance(corr_matrix, dict):
        df_cm = pd.DataFrame(corr_matrix).astype(float)
        target_col = "MedHouseVal" if "MedHouseVal" in df_cm.columns else (df_cm.columns[-1] if len(df_cm.columns) > 0 else None)
        if target_col and target_col in df_cm.columns:
            corr_target = df_cm[target_col].drop(target_col).to_dict()

if corr_target:
    if isinstance(corr_target, dict):
        df_tc = pd.DataFrame({
            tr("feature"): list(corr_target.keys()),
            tr("correlation"): list(corr_target.values()),
        }).sort_values(tr("correlation"), key=abs, ascending=False).head(8)
        fig_corr = px.bar(
            df_tc, x=tr("correlation"), y=tr("feature"),
            orientation="h", title=tr("top_correlations"),
            color=tr("correlation"), color_continuous_scale="RdBu_r",
            range_color=[-1, 1],
        )
        fig_corr.update_layout(height=350)
        st.plotly_chart(fig_corr, use_container_width=True)
    elif isinstance(corr_target, list):
        df_tc = pd.DataFrame(corr_target)
        if not df_tc.empty:
            st.dataframe(df_tc, use_container_width=True)
else:
    st.info(tr("no_data"))

# ── Row 4: Statistical Tests Summary ───────────────────────────
st.subheader(f"🧪 {tr('stats_summary')}")
if stats:
    rows = []
    sh = stats.get("shapiro", stats.get("shapiro_wilk", {}))
    dw = stats.get("durbin_watson", stats.get("dw", {}))
    bp = stats.get("breusch_pagan", stats.get("bp", {}))
    fr = stats.get("friedman", stats.get("friedman_test", {}))

    if sh:
        pv = sh.get("p_value", sh.get("pvalue", sh.get("p", 1)))
        rows.append({tr("test"): tr("shapiro"), tr("result"): tr("normal") if isinstance(pv, (int, float)) and pv > 0.05 else tr("not_normal")})
    if dw:
        dv = dw.get("statistic", dw.get("stat", dw.get("dw", 2)))
        if isinstance(dv, (int, float)):
            dw_r = tr("no_auto") if 1.5 <= dv <= 2.5 else "Auto: " + f"{dv:.2f}"
        else:
            dw_r = "N/A"
        rows.append({tr("test"): tr("dw"), tr("result"): dw_r})
    if bp:
        pv = bp.get("p_value", bp.get("pvalue", bp.get("p", 1)))
        rows.append({tr("test"): tr("bp"), tr("result"): tr("homo") if isinstance(pv, (int, float)) and pv > 0.05 else tr("hetero")})
    if fr:
        pv = fr.get("p_value", fr.get("pvalue", fr.get("p", 1)))
        rows.append({tr("test"): tr("friedman"), tr("result"): tr("sig_diff") if isinstance(pv, (int, float)) and pv < 0.05 else tr("no_sig")})

    if rows:
        df_stats = pd.DataFrame(rows)
        st.dataframe(df_stats, use_container_width=True)
    else:
        st.info(tr("no_data"))
else:
    st.info(tr("no_data"))
