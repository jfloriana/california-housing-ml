import streamlit as st
import pandas as pd
from utils import api_client

if not st.session_state.get("authenticated"):
    st.warning("Please login")
    st.stop()

LANG = {
    "es": {
        "title": "Pronosticar con Modelo Entrenado",
        "subtitle": "Ingrese los valores de las variables para pronosticar",
        "predict": "Pronosticar",
        "result": "Valor Pronosticado",
        "result_desc": "El valor estimado es:",
        "history": "Historial de Pronósticos",
        "no_history": "No hay pronósticos aún",
        "loading_error": "Error al realizar el pronóstico",
        "input_features": "Variables de Entrada",
        "prediction_date": "Fecha",
        "clear_history": "Limpiar Historial",
        "prediction_help": "Complete todos los campos para pronosticar",
        "no_data": "Primero sube un dataset en EDA y entrena los modelos.",
        "model_used": "Modelo usado",
    },
    "en": {
        "title": "Forecast with Trained Model",
        "subtitle": "Enter variable values to forecast",
        "predict": "Forecast",
        "result": "Forecasted Value",
        "result_desc": "The estimated value is:",
        "history": "Forecast History",
        "no_history": "No forecasts yet",
        "loading_error": "Error making forecast",
        "input_features": "Input Variables",
        "prediction_date": "Date",
        "clear_history": "Clear History",
        "prediction_help": "Fill all fields to forecast",
        "no_data": "First upload a dataset in EDA and train the models.",
        "model_used": "Model used",
    },
}

def tr(key):
    return LANG.get(st.session_state.language, LANG["es"]).get(key, key)

lang = st.session_state.language
st.title(tr("title"))
st.markdown(f"*{tr('subtitle')}*")
st.markdown("---")

feature_names = st.session_state.get("feature_names")
if not feature_names:
    st.info(tr("no_data"))
    st.stop()

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

col1, col2 = st.columns([3, 2])

with col1:
    st.subheader(tr("input_features"))
    feature_values = {}
    with st.form("prediction_form"):
        cols = st.columns(2)
        for i, name in enumerate(feature_names):
            with cols[i % 2]:
                feature_values[name] = st.number_input(
                    name, value=0.0, step=0.01, format="%.4f"
                )
        st.markdown("---")
        submitted = st.form_submit_button(
            tr("predict"), use_container_width=True, type="primary"
        )

with col2:
    st.subheader(tr("result"))
    st.caption(tr("prediction_help"))

    if submitted:
        try:
            with st.spinner(tr("predict") + "..."):
                response = api_client.predict(feature_values)
            prediction = response.get("prediction", response.get("value", response.get("price", 0)))
            model_used = response.get("model_used", "CNN-LSTM")

            st.markdown(
                f"""
                <div style="background:#1E3A5F;padding:25px;border-radius:10px;text-align:center;">
                    <p style="color:#A0C4FF;margin:0;font-size:14px;">{tr('result_desc')}</p>
                    <h1 style="color:white;margin:10px 0;font-size:36px;">{prediction:,.4f}</h1>
                    <p style="color:#A0C4FF;margin:0;font-size:12px;">{tr('model_used')}: {model_used}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            from datetime import datetime
            st.session_state.prediction_history.append({
                tr("prediction_date"): datetime.now().strftime("%Y-%m-%d %H:%M"),
                **feature_values,
                tr("result"): prediction,
            })

        except Exception as e:
            st.error(f"{tr('loading_error')}: {e}")

st.markdown("---")
with st.expander(f"📜 {tr('history')} ({len(st.session_state.prediction_history)})"):
    if st.session_state.prediction_history:
        if st.button(tr("clear_history")):
            st.session_state.prediction_history = []
            st.rerun()
        df_hist = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.info(tr("no_history"))
