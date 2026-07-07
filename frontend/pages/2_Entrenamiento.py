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
        "title": "Entrenamiento de Modelos",
        "subtitle": "Comparación de 5 modelos de regresión",
        "comparison_table": "Tabla Comparativa de Modelos",
        "bar_chart": "Comparación Visual R² y RMSE",
        "best_model": "Mejor Modelo",
        "best_model_details": "Detalles del mejor modelo",
        "training_history": "Historial de Entrenamiento",
        "loss_curves": "Curvas de Pérdida",
        "interpretation": "Interpretación de Resultados",
        "no_data": "No se pudieron cargar las métricas de entrenamiento",
        "loading_error": "Error al cargar métricas de entrenamiento",
        "model": "Modelo",
        "mse": "MSE",
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "R²",
        "time": "Tiempo (s)",
        "params": "Parámetros",
        "metric": "Métrica",
        "value": "Valor",
        "epoch": "Época",
        "loss": "Pérdida",
        "val_loss": "Pérdida de Validación",
        "interpretation_text": "El mejor modelo es {best_model} con un R² de {best_r2:.4f} y RMSE de {best_rmse:.4f}. Comparando los {model_count} modelos, se observa que {best_model} supera a los demás en {metric_name}.",
    },
    "en": {
        "title": "Model Training",
        "subtitle": "Comparison of 5 regression models",
        "comparison_table": "Model Comparison Table",
        "bar_chart": "Visual Comparison R² and RMSE",
        "best_model": "Best Model",
        "best_model_details": "Best model details",
        "training_history": "Training History",
        "loss_curves": "Loss Curves",
        "interpretation": "Results Interpretation",
        "no_data": "Could not load training metrics",
        "loading_error": "Error loading training metrics",
        "model": "Model",
        "mse": "MSE",
        "rmse": "RMSE",
        "mae": "MAE",
        "r2": "R²",
        "time": "Time (s)",
        "params": "Parameters",
        "metric": "Metric",
        "value": "Value",
        "epoch": "Epoch",
        "loss": "Loss",
        "val_loss": "Validation Loss",
        "interpretation_text": "The best model is {best_model} with R² of {best_r2:.4f} and RMSE of {best_rmse:.4f}. Comparing all {model_count} models, {best_model} outperforms the others in {metric_name}.",
    },
}

def tr(key):
    return LANG.get(st.session_state.language, LANG["es"]).get(key, key)

lang = st.session_state.language
st.title(tr("title"))
st.markdown(f"*{tr('subtitle')}*")
st.markdown("---")

try:
    with st.spinner("Loading training metrics..."):
        data = api_client.get_training_metrics()
except Exception as e:
    st.error(f"{tr('loading_error')}: {e}")
    st.stop()

models = data.get("models", data.get("results", []))

# ── Section 1: Comparison Table ────────────────────────────────
st.header(tr("comparison_table"))
if models:
    df_models = pd.DataFrame(models)
    metric_cols = [tr("model"), tr("mse"), tr("rmse"), tr("mae"), tr("r2"), tr("time"), tr("params")]
    col_map = {
        "model": "model",
        "name": "model",
        "MSE": tr("mse"),
        "RMSE": tr("rmse"),
        "MAE": tr("mae"),
        "R2": tr("r2"),
        "R²": tr("r2"),
        "r2": tr("r2"),
        "training_time": tr("time"),
        "time": tr("time"),
        "params": tr("params"),
        "parameters": tr("params"),
    }
    df_display = df_models.rename(columns=col_map)
    keep_cols = [c for c in [tr("model"), tr("mse"), tr("rmse"), tr("mae"), tr("r2"), tr("time"), tr("params")] if c in df_display.columns]
    df_display = df_display[keep_cols]

    st.dataframe(
        df_display.style.highlight_max(subset=[tr("r2")], color="lightgreen")
        .highlight_min(subset=[tr("rmse"), tr("mse"), tr("mae")], color="lightcoral"),
        use_container_width=True,
    )
else:
    st.info(tr("no_data"))

# ── Section 2: Bar Chart Comparison ────────────────────────────
st.header(tr("bar_chart"))
if models:
    df_chart = pd.DataFrame(models)
    name_col = next((c for c in ["model_name", "name", "model"] if c in df_chart.columns), None)
    r2_col = next((c for c in ["R2", "R²", "r2"] if c in df_chart.columns), None)
    rmse_col = next((c for c in ["RMSE", "rmse"] if c in df_chart.columns), None)

    if name_col and r2_col and rmse_col:
        df_plot = df_chart[[name_col, r2_col, rmse_col]].melt(
            id_vars=[name_col], var_name=tr("metric"), value_name=tr("value")
        )
        fig_bar = px.bar(
            df_plot,
            x=name_col,
            y=tr("value"),
            color=tr("metric"),
            barmode="group",
            title=f"R² vs RMSE por {tr('model')}",
            color_discrete_sequence=["#2E91E5", "#E76376"],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ── Section 3: Best Model Details ──────────────────────────────
st.header(tr("best_model"))
best_model = data.get("best_model", data.get("best", {}))
if not best_model and models:
    df_temp = pd.DataFrame(models)
    r2_c = next((c for c in ["R2", "R²", "r2"] if c in df_temp.columns), None)
    if r2_c:
        best_idx = df_temp[r2_c].idxmax()
        best_model = df_temp.iloc[best_idx].to_dict()

if best_model:
    best_name = best_model.get("model_name") or best_model.get("model") or best_model.get("name", tr("best_model"))
    st.subheader(f"🏆 {best_name}")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        r2_val = best_model.get("R2") or best_model.get("R²") or best_model.get("r2", "N/A")
        st.metric("R²", f"{r2_val:.4f}" if isinstance(r2_val, (int, float)) else r2_val)
    with c2:
        rmse_val = best_model.get("RMSE") or best_model.get("rmse", "N/A")
        st.metric("RMSE", f"{rmse_val:.4f}" if isinstance(rmse_val, (int, float)) else rmse_val)
    with c3:
        mae_val = best_model.get("MAE") or best_model.get("mae", "N/A")
        st.metric("MAE", f"{mae_val:.4f}" if isinstance(mae_val, (int, float)) else mae_val)
    with c4:
        mse_val = best_model.get("MSE") or best_model.get("mse", "N/A")
        st.metric("MSE", f"{mse_val:.4f}" if isinstance(mse_val, (int, float)) else mse_val)

    # Show params
    params = best_model.get("params") or best_model.get("parameters", {})
    if isinstance(params, dict) and params:
        with st.expander(tr("best_model_details")):
            st.json(params)
    elif isinstance(params, str):
        with st.expander(tr("best_model_details")):
            st.text(params)

# ── Section 4: Training History ─────────────────────────────────
st.header(tr("training_history"))
history = data.get("history", data.get("training_history", data.get("loss_curves", [])))
if history:
    if isinstance(history, list) and len(history) > 0:
        df_hist = pd.DataFrame(history)
        if "epoch" in df_hist.columns or "loss" in df_hist.columns:
            x_col = "epoch" if "epoch" in df_hist.columns else df_hist.columns[0]
            y_cols = [c for c in df_hist.columns if c != x_col]
            fig_hist = go.Figure()
            for yc in y_cols:
                fig_hist.add_trace(go.Scatter(
                    x=df_hist[x_col], y=df_hist[yc],
                    mode="lines+markers", name=yc,
                ))
            fig_hist.update_layout(
                title=tr("loss_curves"),
                xaxis_title=tr("epoch"),
                yaxis_title=tr("loss"),
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.dataframe(df_hist, use_container_width=True)
    elif isinstance(history, dict):
        df_hist = pd.DataFrame(history)
        if not df_hist.empty:
            st.dataframe(df_hist, use_container_width=True)
else:
    st.info("No training history available.")

# ── Section 5: Interpretation ──────────────────────────────────
st.header(tr("interpretation"))
if models:
    df_temp = pd.DataFrame(models)
    best_name_val = best_model.get("model_name") or best_model.get("model") or best_model.get("name", "N/A")
    r2_col = next((c for c in ["R2", "R²", "r2"] if c in df_temp.columns), None)
    rmse_col = next((c for c in ["RMSE", "rmse"] if c in df_temp.columns), None)
    best_r2 = best_model.get(r2_col, 0) if r2_col else 0
    best_rmse = best_model.get(rmse_col, 0) if rmse_col else 0

    st.info(
        tr("interpretation_text").format(
            best_model=best_name_val,
            best_r2=best_r2,
            best_rmse=best_rmse,
            model_count=len(models),
            metric_name="R²",
        )
    )
