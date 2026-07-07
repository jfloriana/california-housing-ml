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
        "title": "Optimización de Hiperparámetros",
        "subtitle": "Búsqueda de la mejor configuración de hiperparámetros",
        "best_params": "Mejores Hiperparámetros",
        "top_configs": "Comparación Top 3 Configuraciones",
        "param_importance": "Importancia de Parámetros",
        "optimization_history": "Historial de Optimización",
        "interpretation": "Interpretación de Resultados",
        "no_data": "No se pudieron cargar los datos de optimización",
        "loading_error": "Error al cargar datos de optimización",
        "parameter": "Parámetro",
        "value": "Valor",
        "score": "Puntuación",
        "rank": "Rango",
        "config": "Configuración",
        "trial": "Intento",
        "interpretation_text": "La mejor configuración de hiperparámetros obtuvo un score de {best_score:.4f}. Los parámetros más importantes fueron {important_params}. Comparado con la configuración base, se logró una mejora del {improvement:.1f}%.",
    },
    "en": {
        "title": "Hyperparameter Tuning",
        "subtitle": "Search for the best hyperparameter configuration",
        "best_params": "Best Hyperparameters",
        "top_configs": "Top 3 Configurations Comparison",
        "param_importance": "Parameter Importance",
        "optimization_history": "Optimization History",
        "interpretation": "Results Interpretation",
        "no_data": "Could not load hyperparameter tuning data",
        "loading_error": "Error loading hyperparameter tuning data",
        "parameter": "Parameter",
        "value": "Value",
        "score": "Score",
        "rank": "Rank",
        "config": "Configuration",
        "trial": "Trial",
        "interpretation_text": "The best hyperparameter configuration achieved a score of {best_score:.4f}. The most important parameters were {important_params}. Compared to the baseline configuration, an improvement of {improvement:.1f}% was achieved.",
    },
}

def tr(key):
    return LANG.get(st.session_state.language, LANG["es"]).get(key, key)

lang = st.session_state.language
st.title(tr("title"))
st.markdown(f"*{tr('subtitle')}*")
st.markdown("---")

cache = st.session_state.api_metrics_cache
if "tuning" not in cache:
    try:
        with st.spinner("Loading hyperparameter tuning data..."):
            cache["tuning"] = api_client.get_hyperparameter_tuning()
    except Exception as e:
        st.error(f"{tr('loading_error')}: {e}")
        st.stop()
data = cache["tuning"]

tuning_raw = data.get("hyperparameter_tuning", data.get("tuning", data.get("results", data.get("trials", []))))
if isinstance(tuning_raw, list):
    tuning_dict = {"results": tuning_raw}
elif isinstance(tuning_raw, dict):
    tuning_dict = tuning_raw
else:
    tuning_dict = {}

# ── Section 1: Best Hyperparameters ────────────────────────────
st.header(tr("best_params"))
best_params = tuning_dict.get("best_params", data.get("best_params", data.get("best_parameters", data.get("best", {}))))
if best_params:
    if isinstance(best_params, dict):
        bp_items = [k for k in best_params.keys() if k not in ("score", "Score", "value", "Value")]
        df_bp = pd.DataFrame([
            {tr("parameter"): k, tr("value"): v}
            for k, v in best_params.items()
            if k in bp_items or k not in ("score", "Score")
        ])
        st.dataframe(df_bp, use_container_width=True)

        best_score = best_params.get("score") or best_params.get("Score") or best_params.get("best_score")
        if best_score:
            st.metric(tr("score"), f"{best_score:.4f}" if isinstance(best_score, (int, float)) else best_score)
    elif isinstance(best_params, list):
        df_bp = pd.DataFrame(best_params)
        st.dataframe(df_bp, use_container_width=True)
else:
    st.info(tr("no_data"))

# ── Section 2: Top 3 Configurations ────────────────────────────
st.header(tr("top_configs"))
all_results = tuning_dict.get("results", tuning_dict.get("trials", data.get("results", data.get("trials", data.get("configurations", [])))))
if all_results:
    df_all = pd.DataFrame(all_results)
    score_col = next((c for c in ["score", "Score", "mean_test_score", "value"] if c in df_all.columns), None)
    if score_col and len(df_all) > 0:
        df_sorted = df_all.sort_values(score_col, ascending=False).head(3)
        st.dataframe(df_sorted.style.highlight_max(subset=[score_col], color="lightgreen"), use_container_width=True)

        # Bar chart of top 3
        df_sorted["rank_label"] = [f"{tr('config')} {i+1}" for i in range(len(df_sorted))]
        fig_top = px.bar(
            df_sorted, x="rank_label", y=score_col,
            title=f"{tr('top_configs')} - {tr('score')}",
            color=score_col, color_continuous_scale="viridis",
        )
        st.plotly_chart(fig_top, use_container_width=True)
    else:
        st.dataframe(df_all.head(3), use_container_width=True)

# ── Section 3: Parameter Importance ────────────────────────────
st.header(tr("param_importance"))
importance = tuning_dict.get("importance", data.get("importance", data.get("param_importance", data.get("feature_importance", {}))))
if importance:
    if isinstance(importance, dict):
        df_imp = pd.DataFrame([
            {tr("parameter"): k, tr("value"): v} for k, v in importance.items()
        ]).sort_values(tr("value"), ascending=False)
        fig_imp = px.bar(
            df_imp, x=tr("parameter"), y=tr("value"),
            title=tr("param_importance"),
            color=tr("value"), color_continuous_scale="blues",
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    elif isinstance(importance, list):
        df_imp = pd.DataFrame(importance)
        st.dataframe(df_imp, use_container_width=True)
        if len(df_imp.columns) >= 2:
            fig_imp = px.bar(df_imp, x=df_imp.columns[0], y=df_imp.columns[1], title=tr("param_importance"))
            st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.info("No parameter importance data available.")

# ── Section 4: Optimization History ────────────────────────────
st.header(tr("optimization_history"))
history = tuning_dict.get("history", data.get("history", data.get("optimization_history", data.get("search_history", []))))
if history:
    if isinstance(history, list):
        df_hist = pd.DataFrame(history)
        score_c = next((c for c in ["score", "Score", "value", "objective"] if c in df_hist.columns), df_hist.columns[-1] if len(df_hist.columns) > 0 else None)
        trial_c = next((c for c in ["trial", "Trial", "iteration", "Iteration", "step"] if c in df_hist.columns), None)
        if trial_c and score_c:
            fig_hist = px.line(
                df_hist, x=trial_c, y=score_c,
                markers=True, title=tr("optimization_history"),
            )
            fig_hist.update_layout(xaxis_title=tr("trial"), yaxis_title=tr("score"))
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.dataframe(df_hist, use_container_width=True)
    elif isinstance(history, dict):
        st.dataframe(pd.DataFrame([history]), use_container_width=True)
else:
    st.info("No optimization history available.")

# ── Section 5: Interpretation ──────────────────────────────────
st.header(tr("interpretation"))
best_score_val = 0
if best_params and isinstance(best_params, dict):
    best_score_val = best_params.get("score") or best_params.get("Score") or 0
elif isinstance(best_params, (int, float)):
    best_score_val = best_params

important_params_text = "N/A"
if importance:
    if isinstance(importance, dict):
        sorted_imp = sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        important_params_text = ", ".join([f"{k}" for k, v in sorted_imp])
    elif isinstance(importance, list) and len(importance) > 0:
        df_imp_t = pd.DataFrame(importance)
        if len(df_imp_t.columns) >= 2:
            top_imp = df_imp_t.sort_values(df_imp_t.columns[1], ascending=False).head(3)
            important_params_text = ", ".join(top_imp.iloc[:, 0].astype(str).tolist())

improvement = data.get("improvement", data.get("improvement_percent", 0))
if not isinstance(improvement, (int, float)):
    improvement = 0

st.info(
    tr("interpretation_text").format(
        best_score=best_score_val if isinstance(best_score_val, (int, float)) else 0,
        important_params=important_params_text,
        improvement=improvement,
    )
)
