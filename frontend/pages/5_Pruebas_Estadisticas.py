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
        "title": "Pruebas Estadísticas",
        "subtitle": "Validación estadística de los modelos de regresión",
        "shapiro": "Prueba de Normalidad (Shapiro-Wilk)",
        "shapiro_desc": "Evalúa si los residuos siguen una distribución normal",
        "durbin_watson": "Prueba de Autocorrelación (Durbin-Watson)",
        "dw_desc": "Detecta autocorrelación en los residuos",
        "breusch_pagan": "Prueba de Heterocedasticidad (Breusch-Pagan)",
        "bp_desc": "Evalúa si la varianza de los residuos es constante",
        "friedman": "Prueba de Comparación de Modelos (Friedman)",
        "friedman_desc": "Compara el rendimiento de múltiples modelos",
        "pairwise": "Comparaciones Pareadas",
        "combined_interpretation": "Interpretación Combinada",
        "no_data": "No se pudieron cargar los datos de pruebas estadísticas",
        "loading_error": "Error al cargar datos de pruebas estadísticas",
        "statistic": "Estadístico",
        "p_value": "Valor p",
        "result": "Resultado",
        "normal": "Normal",
        "not_normal": "No normal",
        "no_autocorrelation": "Sin autocorrelación",
        "positive_autocorrelation": "Autocorrelación positiva",
        "negative_autocorrelation": "Autocorrelación negativa",
        "homoscedastic": "Homocedástico",
        "heteroscedastic": "Heterocedástico",
        "significant_difference": "Diferencias significativas",
        "no_significant_difference": "Sin diferencias significativas",
        "model_1": "Modelo 1",
        "model_2": "Modelo 2",
        "statistic_label": "Estadístico",
        "interpretation_text": "Las pruebas estadísticas indican que los residuos {normality_text}. La prueba de Durbin-Watson ({dw_stat:.2f}) sugiere {dw_text}. La prueba de Breusch-Pagan ({bp_stat:.2f}, p={bp_p:.4f}) indica {bp_text}. La prueba de Friedman ({f_stat:.2f}, p={f_p:.4f}) muestra {friedman_text} entre los modelos.",
        "are_normal": "siguen una distribución normal",
        "are_not_normal": "NO siguen una distribución normal",
        "inconclusive_autocorrelation": "autocorrelación inconclusa",
    },
    "en": {
        "title": "Statistical Tests",
        "subtitle": "Statistical validation of regression models",
        "shapiro": "Normality Test (Shapiro-Wilk)",
        "shapiro_desc": "Evaluates whether residuals follow a normal distribution",
        "durbin_watson": "Autocorrelation Test (Durbin-Watson)",
        "dw_desc": "Detects autocorrelation in residuals",
        "breusch_pagan": "Heteroscedasticity Test (Breusch-Pagan)",
        "bp_desc": "Evaluates whether residual variance is constant",
        "friedman": "Model Comparison Test (Friedman)",
        "friedman_desc": "Compares performance of multiple models",
        "pairwise": "Pairwise Comparisons",
        "combined_interpretation": "Combined Interpretation",
        "no_data": "Could not load statistical tests data",
        "loading_error": "Error loading statistical tests data",
        "statistic": "Statistic",
        "p_value": "P-value",
        "result": "Result",
        "normal": "Normal",
        "not_normal": "Not normal",
        "no_autocorrelation": "No autocorrelation",
        "positive_autocorrelation": "Positive autocorrelation",
        "negative_autocorrelation": "Negative autocorrelation",
        "homoscedastic": "Homoscedastic",
        "heteroscedastic": "Heteroscedastic",
        "significant_difference": "Significant differences",
        "no_significant_difference": "No significant differences",
        "model_1": "Model 1",
        "model_2": "Model 2",
        "statistic_label": "Statistic",
        "interpretation_text": "Statistical tests indicate that the residuals {normality_text}. The Durbin-Watson test ({dw_stat:.2f}) suggests {dw_text}. The Breusch-Pagan test ({bp_stat:.2f}, p={bp_p:.4f}) indicates {bp_text}. The Friedman test ({f_stat:.2f}, p={f_p:.4f}) shows {friedman_text} between models.",
        "are_normal": "follow a normal distribution",
        "are_not_normal": "do NOT follow a normal distribution",
        "inconclusive_autocorrelation": "inconclusive autocorrelation",
    },
}

def tr(key):
    return LANG.get(st.session_state.language, LANG["es"]).get(key, key)

lang = st.session_state.language
st.title(tr("title"))
st.markdown(f"*{tr('subtitle')}*")
st.markdown("---")

try:
    with st.spinner("Loading statistical test data..."):
        data = api_client.get_statistical_tests()
except Exception as e:
    st.error(f"{tr('loading_error')}: {e}")
    st.stop()

# ── Section 1: Shapiro-Wilk ────────────────────────────────────
st.header(tr("shapiro"))
st.caption(tr("shapiro_desc"))
shapiro = data.get("shapiro", data.get("shapiro_wilk", data.get("shapiro_test", {})))
if shapiro:
    stat_s = shapiro.get("statistic", shapiro.get("stat", shapiro.get("w", "N/A")))
    pval_s = shapiro.get("p_value", shapiro.get("pvalue", shapiro.get("p", "N/A")))
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(tr("statistic"), f"{stat_s:.4f}" if isinstance(stat_s, (int, float)) else stat_s)
    with c2:
        st.metric(tr("p_value"), f"{pval_s:.4f}" if isinstance(pval_s, (int, float)) else pval_s)
    with c3:
        is_normal = (isinstance(pval_s, (int, float)) and pval_s > 0.05)
        st.metric(tr("result"), tr("normal") if is_normal else tr("not_normal"),
                  delta="✓" if is_normal else "✗")
else:
    st.info("Shapiro-Wilk test data not available.")

# ── Section 2: Durbin-Watson ────────────────────────────────────
st.header(tr("durbin_watson"))
st.caption(tr("dw_desc"))
dw = data.get("durbin_watson", data.get("dw", data.get("durbin_watson_test", {})))
if dw:
    dw_stat = dw.get("statistic", dw.get("stat", dw.get("dw", "N/A")))
    dw_val = dw_stat if isinstance(dw_stat, (int, float)) else 0
    if isinstance(dw_val, (int, float)):
        if dw_val < 1.5:
            dw_result = tr("positive_autocorrelation")
        elif dw_val > 2.5:
            dw_result = tr("negative_autocorrelation")
        else:
            dw_result = tr("no_autocorrelation")
    else:
        dw_result = "N/A"

    c1, c2 = st.columns(2)
    with c1:
        st.metric(tr("statistic"), f"{dw_stat:.4f}" if isinstance(dw_stat, (int, float)) else dw_stat)
    with c2:
        st.metric(tr("result"), dw_result)
else:
    st.info("Durbin-Watson test data not available.")

# ── Section 3: Breusch-Pagan ────────────────────────────────────
st.header(tr("breusch_pagan"))
st.caption(tr("bp_desc"))
bp = data.get("breusch_pagan", data.get("bp", data.get("breusch_pagan_test", {})))
if bp:
    bp_stat = bp.get("statistic", bp.get("stat", bp.get("lm", "N/A")))
    bp_pval = bp.get("p_value", bp.get("pvalue", bp.get("p", "N/A")))
    is_homo = (isinstance(bp_pval, (int, float)) and bp_pval > 0.05)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(tr("statistic"), f"{bp_stat:.4f}" if isinstance(bp_stat, (int, float)) else bp_stat)
    with c2:
        st.metric(tr("p_value"), f"{bp_pval:.4f}" if isinstance(bp_pval, (int, float)) else bp_pval)
    with c3:
        st.metric(tr("result"), tr("homoscedastic") if is_homo else tr("heteroscedastic"),
                  delta="✓" if is_homo else "✗")
else:
    st.info("Breusch-Pagan test data not available.")

# ── Section 4: Friedman Test ────────────────────────────────────
st.header(tr("friedman"))
st.caption(tr("friedman_desc"))
friedman = data.get("friedman", data.get("friedman_test", {}))
if friedman:
    f_stat = friedman.get("statistic", friedman.get("stat", "N/A"))
    f_pval = friedman.get("p_value", friedman.get("pvalue", friedman.get("p", "N/A")))
    is_sig = (isinstance(f_pval, (int, float)) and f_pval < 0.05)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(tr("statistic"), f"{f_stat:.4f}" if isinstance(f_stat, (int, float)) else f_stat)
    with c2:
        st.metric(tr("p_value"), f"{f_pval:.4f}" if isinstance(f_pval, (int, float)) else f_pval)
    with c3:
        st.metric(tr("result"), tr("significant_difference") if is_sig else tr("no_significant_difference"),
                  delta="⚡" if is_sig else "—")
else:
    st.info("Friedman test data not available.")

# ── Section 5: Pairwise Comparisons ────────────────────────────
st.header(tr("pairwise"))
pairwise = data.get("pairwise", data.get("pairwise_comparisons", data.get("posthoc", [])))
if pairwise:
    if isinstance(pairwise, list):
        df_pair = pd.DataFrame(pairwise)
        st.dataframe(df_pair.style.format("{:.4f}" if df_pair.select_dtypes(include="number").shape[1] > 0 else "{}"), use_container_width=True)
    elif isinstance(pairwise, dict):
        df_pair = pd.DataFrame(pairwise)
        if not df_pair.empty:
            fig_heat_pair = go.Figure(data=go.Heatmap(
                z=df_pair.values.astype(float),
                x=df_pair.columns.tolist(),
                y=df_pair.index.tolist(),
                colorscale="RdBu_r",
                text=[[f"{v:.3f}" for v in row] for row in df_pair.values],
                texttemplate="%{text}",
            ))
            fig_heat_pair.update_layout(title=tr("pairwise"), height=400)
            st.plotly_chart(fig_heat_pair, use_container_width=True)
else:
    # Try combined tests table
    tests_table = data.get("tests", data.get("all_tests", []))
    if tests_table:
        df_tests = pd.DataFrame(tests_table)
        st.dataframe(df_tests, use_container_width=True)
    else:
        st.info("No pairwise comparison data available.")

# ── Section 6: Combined Interpretation ──────────────────────────
st.header(tr("combined_interpretation"))
normality_text = tr("are_normal")
if shapiro:
    pv_s = shapiro.get("p_value", shapiro.get("pvalue", shapiro.get("p", 1)))
    if isinstance(pv_s, (int, float)) and pv_s <= 0.05:
        normality_text = tr("are_not_normal")

dw_stat_val = 2.0
dw_text = tr("no_autocorrelation")
if dw:
    dw_s = dw.get("statistic", dw.get("stat", dw.get("dw", 2)))
    if isinstance(dw_s, (int, float)):
        dw_stat_val = dw_s
        if dw_s < 1.5:
            dw_text = tr("positive_autocorrelation")
        elif dw_s > 2.5:
            dw_text = tr("negative_autocorrelation")
        else:
            dw_text = tr("no_autocorrelation")

bp_stat_val = 0
bp_p_val = 1
bp_text = tr("homoscedastic")
if bp:
    bp_s = bp.get("statistic", bp.get("stat", bp.get("lm", 0)))
    bp_p = bp.get("p_value", bp.get("pvalue", bp.get("p", 1)))
    if isinstance(bp_s, (int, float)):
        bp_stat_val = bp_s
    if isinstance(bp_p, (int, float)):
        bp_p_val = bp_p
        bp_text = tr("heteroscedastic") if bp_p < 0.05 else tr("homoscedastic")

f_stat_val = 0
f_p_val = 1
friedman_text = tr("no_significant_difference")
if friedman:
    f_s = friedman.get("statistic", friedman.get("stat", 0))
    f_p = friedman.get("p_value", friedman.get("pvalue", friedman.get("p", 1)))
    if isinstance(f_s, (int, float)):
        f_stat_val = f_s
    if isinstance(f_p, (int, float)):
        f_p_val = f_p
        friedman_text = tr("significant_difference") if f_p < 0.05 else tr("no_significant_difference")

st.info(
    tr("interpretation_text").format(
        normality_text=normality_text,
        dw_stat=dw_stat_val,
        dw_text=dw_text,
        bp_stat=bp_stat_val,
        bp_p=bp_p_val,
        bp_text=bp_text,
        f_stat=f_stat_val,
        f_p=f_p_val,
        friedman_text=friedman_text,
    )
)
