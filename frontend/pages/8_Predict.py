import streamlit as st
import pandas as pd
from utils import api_client

if not st.session_state.get("authenticated"):
    st.warning("Please login")
    st.stop()

LANG = {
    "es": {
        "title": "Predecir Valor de Vivienda",
        "subtitle": "Ingrese las características de la vivienda para predecir su valor",
        "medinc": "Ingreso Medio (MedInc)",
        "medinc_help": "Ingreso medio del bloque censal en decenas de miles",
        "houseage": "Antigüedad de la Vivienda (HouseAge)",
        "houseage_help": "Edad media de las viviendas en años",
        "averooms": "Promedio de Habitaciones (AveRooms)",
        "averooms_help": "Número promedio de habitaciones por hogar",
        "avebedrms": "Promedio de Dormitorios (AveBedrms)",
        "avebedrms_help": "Número promedio de dormitorios por hogar",
        "population": "Población",
        "population_help": "Población del bloque censal",
        "aveoccup": "Ocupación Promedio (AveOccup)",
        "aveoccup_help": "Número promedio de ocupantes por hogar",
        "latitude": "Latitud",
        "longitude": "Longitud",
        "predict": "Predecir Valor",
        "result": "Valor Predicho",
        "result_desc": "El valor estimado de la vivienda es:",
        "confidence": "Intervalo de Confianza",
        "confidence_desc": "IC 95%:",
        "history": "Historial de Predicciones",
        "no_history": "No hay predicciones aún",
        "loading_error": "Error al realizar la predicción",
        "input_features": "Características de Entrada",
        "prediction_date": "Fecha",
        "clear_history": "Limpiar Historial",
        "prediction_unit": "USD $",
        "prediction_help": "Complete todos los campos y presione el botón para predecir",
    },
    "en": {
        "title": "Predict Housing Value",
        "subtitle": "Enter housing characteristics to predict value",
        "medinc": "Median Income (MedInc)",
        "medinc_help": "Median income of the census block in tens of thousands",
        "houseage": "Housing Age (HouseAge)",
        "houseage_help": "Average age of houses in years",
        "averooms": "Average Rooms (AveRooms)",
        "averooms_help": "Average number of rooms per household",
        "avebedrms": "Average Bedrooms (AveBedrms)",
        "avebedrms_help": "Average number of bedrooms per household",
        "population": "Population",
        "population_help": "Population of the census block",
        "aveoccup": "Average Occupancy (AveOccup)",
        "aveoccup_help": "Average number of occupants per household",
        "latitude": "Latitude",
        "longitude": "Longitude",
        "predict": "Predict Value",
        "result": "Predicted Value",
        "result_desc": "The estimated housing value is:",
        "confidence": "Confidence Interval",
        "confidence_desc": "95% CI:",
        "history": "Prediction History",
        "no_history": "No predictions yet",
        "loading_error": "Error making prediction",
        "input_features": "Input Features",
        "prediction_date": "Date",
        "clear_history": "Clear History",
        "prediction_unit": "USD $",
        "prediction_help": "Fill all fields and press the button to predict",
    },
}

def tr(key):
    return LANG.get(st.session_state.language, LANG["es"]).get(key, key)

lang = st.session_state.language
st.title(tr("title"))
st.markdown(f"*{tr('subtitle')}*")
st.markdown("---")

# Initialize prediction history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ── Input Form ───────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader(tr("input_features"))
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            medinc = st.slider(tr("medinc"), 0.5, 15.0, 3.87, 0.01, help=tr("medinc_help"))
            houseage = st.slider(tr("houseage"), 1, 52, 28, 1, help=tr("houseage_help"))
            averooms = st.slider(tr("averooms"), 1.0, 20.0, 5.4, 0.1, help=tr("averooms_help"))
            avebedrms = st.slider(tr("avebedrms"), 0.3, 10.0, 1.1, 0.1, help=tr("avebedrms_help"))
        with c2:
            population = st.number_input(tr("population"), 3, 50000, 1425, 1, help=tr("population_help"))
            aveoccup = st.slider(tr("aveoccup"), 0.5, 10.0, 3.1, 0.1, help=tr("aveoccup_help"))
            latitude = st.slider(tr("latitude"), 32.5, 42.0, 34.0, 0.1)
            longitude = st.slider(tr("longitude"), -124.5, -114.0, -118.5, 0.1)

        st.markdown("---")
        submitted = st.form_submit_button(
            tr("predict"), use_container_width=True, type="primary"
        )

with col2:
    st.subheader(tr("result"))
    st.caption(tr("prediction_help"))

    if submitted:
        features = {
            "MedInc": medinc,
            "HouseAge": houseage,
            "AveRooms": averooms,
            "AveBedrms": avebedrms,
            "Population": population,
            "AveOccup": aveoccup,
            "Latitude": latitude,
            "Longitude": longitude,
        }

        try:
            with st.spinner(tr("predict") + "..."):
                response = api_client.predict(features)

            prediction = response.get("prediction", response.get("value", response.get("price", 0)))
            lower_bound = response.get("lower", response.get("lower_bound", response.get("ci_lower", None)))
            upper_bound = response.get("upper", response.get("upper_bound", response.get("ci_upper", None)))
            confidence = response.get("confidence", response.get("confidence_interval", None))

            # Display prediction
            st.markdown(
                f"""
                <div style="background:#1E3A5F;padding:25px;border-radius:10px;text-align:center;">
                    <p style="color:#A0C4FF;margin:0;font-size:14px;">{tr('result_desc')}</p>
                    <h1 style="color:white;margin:10px 0;font-size:36px;">{tr('prediction_unit')}{prediction:,.2f}</h1>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Confidence interval
            if lower_bound is not None and upper_bound is not None:
                st.markdown(
                    f"""
                    <div style="background:#1E5F3A;padding:15px;border-radius:10px;text-align:center;margin-top:10px;">
                        <p style="color:#A0FFC4;margin:0;font-size:14px;">{tr('confidence_desc')}</p>
                        <p style="color:white;margin:5px 0;font-size:20px;">
                            {tr('prediction_unit')}{lower_bound:,.2f} - {tr('prediction_unit')}{upper_bound:,.2f}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif confidence:
                if isinstance(confidence, dict):
                    lo = confidence.get("lower", confidence.get("low", "N/A"))
                    hi = confidence.get("upper", confidence.get("high", "N/A"))
                    st.metric(tr("confidence"), f"{lo} - {hi}")
                else:
                    st.metric(tr("confidence"), str(confidence))

            # Save to history
            from datetime import datetime
            st.session_state.prediction_history.append({
                tr("prediction_date"): datetime.now().strftime("%Y-%m-%d %H:%M"),
                tr("medinc"): medinc,
                tr("houseage"): houseage,
                tr("averooms"): averooms,
                tr("avebedrms"): avebedrms,
                tr("population"): population,
                tr("aveoccup"): aveoccup,
                tr("latitude"): latitude,
                tr("longitude"): longitude,
                tr("result"): prediction,
            })

        except Exception as e:
            st.error(f"{tr('loading_error')}: {e}")

# ── History ──────────────────────────────────────────────────────
st.markdown("---")
with st.expander(f"📜 {tr('history')} ({len(st.session_state.prediction_history)})"):
    if st.session_state.prediction_history:
        if st.button(tr("clear_history")):
            st.session_state.prediction_history = []
            st.rerun()

        df_hist = pd.DataFrame(st.session_state.prediction_history)
        st.dataframe(
            df_hist.style.format({tr("result"): "${:,.2f}"}),
            use_container_width=True,
        )
    else:
        st.info(tr("no_history"))
