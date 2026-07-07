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

# ── Build config list from API data ─────────────────────────────────
all_results = tuning_dict.get("results", data.get("results", data.get("trials", [])))

if all_results:
    # Find best by best_r2
    best_idx = 0
    best_r2_val = 0
    for i, r in enumerate(all_results):
        r2_candidate = r.get("best_r2", r.get("r2", 0))
        if r2_candidate > best_r2_val:
            best_r2_val = r2_candidate
            best_idx = i
    best_entry = all_results[best_idx]

    # ── Section 1: Best Hyperparameters ────────────────────────────
    st.header(tr("best_params"))
    params = best_entry.get("params", {})
    if isinstance(params, dict) and params:
        df_bp = pd.DataFrame([
            {tr("parameter"): k, tr("value"): str(v)} for k, v in params.items()
        ])
        st.dataframe(df_bp, use_container_width=True)
        best_score_val = best_entry.get("best_r2", best_entry.get("r2", 0))
        st.metric(f"{tr('score')} (R²)", f"{best_score_val:.4f}")

    # ── Section 2: All Configurations ──────────────────────────────
    st.header(tr("top_configs"))
    df_all = pd.DataFrame(all_results)
    display_cols = [c for c in ["model_name", "best_mse", "best_r2"] if c in df_all.columns]
    if display_cols:
        df_show = df_all[display_cols].sort_values("best_r2", ascending=False)
        st.dataframe(df_show.style.highlight_max(subset=["best_r2"], color="lightgreen")
                     .highlight_min(subset=["best_mse"], color="lightcoral"),
                     use_container_width=True)
    else:
        st.dataframe(df_all, use_container_width=True)

    # ── Section 3: Interpretation ──────────────────────────────────
    st.header(tr("interpretation"))
    best_name = best_entry.get("model_name", "N/A")
    st.info(
        tr("interpretation_text").format(
            best_score=best_r2_val,
            important_params=", ".join(params.keys()) if isinstance(params, dict) else "N/A",
            improvement=0,
        )
    )
else:
    st.info(tr("no_data"))
    st.header(tr("interpretation"))
    st.info(tr("interpretation_text").format(best_score=0, important_params="N/A", improvement=0))
