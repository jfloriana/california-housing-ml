import streamlit as st
import requests
import io
from utils import api_client

if not st.session_state.get("authenticated"):
    st.warning("Please login")
    st.stop()

LANG = {
    "es": {
        "title": "Generación de Reportes",
        "subtitle": "Descargue reportes detallados del análisis completo",
        "language_selection": "Selección de Idioma",
        "download_pdf": "Descargar Reporte PDF",
        "download_word": "Descargar Reporte Word",
        "download_excel": "Descargar Reporte Excel",
        "preview": "Vista Previa del Contenido",
        "sections": "Secciones del Reporte",
        "sections_list": [
            "Resumen Ejecutivo",
            "Análisis Exploratorio de Datos",
            "Entrenamiento de Modelos",
            "Validación Cruzada",
            "Optimización de Hiperparámetros",
            "Pruebas Estadísticas",
            "Conclusiones y Recomendaciones",
        ],
        "preview_text": "Este reporte incluye análisis completo del dataset California Housing, incluyendo EDA, entrenamiento de 5 modelos de regresión, validación cruzada, optimización de hiperparámetros y pruebas estadísticas.",
        "generating": "Generando reporte...",
        "success": "Reporte generado correctamente",
        "error": "Error al generar el reporte",
        "no_backend": "No se pudo conectar con el backend para generar reportes",
        "select_language": "Seleccione el idioma del reporte",
    },
    "en": {
        "title": "Report Generation",
        "subtitle": "Download detailed reports of the complete analysis",
        "language_selection": "Language Selection",
        "download_pdf": "Download PDF Report",
        "download_word": "Download Word Report",
        "download_excel": "Download Excel Report",
        "preview": "Content Preview",
        "sections": "Report Sections",
        "sections_list": [
            "Executive Summary",
            "Exploratory Data Analysis",
            "Model Training",
            "Cross Validation",
            "Hyperparameter Tuning",
            "Statistical Tests",
            "Conclusions and Recommendations",
        ],
        "preview_text": "This report includes a complete analysis of the California Housing dataset, including EDA, training of 5 regression models, cross validation, hyperparameter tuning, and statistical tests.",
        "generating": "Generating report...",
        "success": "Report generated successfully",
        "error": "Error generating report",
        "no_backend": "Could not connect to the backend to generate reports",
        "select_language": "Select report language",
    },
}

def tr(key):
    val = LANG.get(st.session_state.language, LANG["es"]).get(key, key)
    return val

lang = st.session_state.language
st.title(tr("title"))
st.markdown(f"*{tr('subtitle')}*")
st.markdown("---")

# Language selector
st.subheader(tr("language_selection"))
report_lang = st.radio(
    tr("select_language"),
    options=["es", "en"],
    format_func=lambda x: "Español" if x == "es" else "English",
    index=0 if lang == "es" else 1,
    horizontal=True,
)

st.markdown("---")

# ── Download buttons ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

def fetch_report(format_name, language):
    try:
        url = api_client.generate_report_url(format_name, language)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        st.error(f"{tr('error')}: {e}")
        return None

with col1:
    with st.container():
        st.markdown("##### 📄 PDF")
        st.markdown("Portable Document Format")
        pdf_bytes = fetch_report("pdf", report_lang)
        if pdf_bytes:
            st.download_button(
                label=tr("download_pdf"),
                data=pdf_bytes,
                file_name=f"california_housing_report_{report_lang}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.warning(tr("no_backend"))

with col2:
    with st.container():
        st.markdown("##### 📝 Word")
        st.markdown("Microsoft Word format")
        docx_bytes = fetch_report("word", report_lang)
        if docx_bytes:
            st.download_button(
                label=tr("download_word"),
                data=docx_bytes,
                file_name=f"california_housing_report_{report_lang}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )
        else:
            st.warning(tr("no_backend"))

with col3:
    with st.container():
        st.markdown("##### 📊 Excel")
        st.markdown("Microsoft Excel format")
        xlsx_bytes = fetch_report("excel", report_lang)
        if xlsx_bytes:
            st.download_button(
                label=tr("download_excel"),
                data=xlsx_bytes,
                file_name=f"california_housing_report_{report_lang}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        else:
            st.warning(tr("no_backend"))

st.markdown("---")

# ── Preview ──────────────────────────────────────────────────────
st.header(tr("preview"))
st.info(tr("preview_text"))

st.subheader(tr("sections"))
sections = tr("sections_list")
for i, section in enumerate(sections, 1):
    st.markdown(f"**{i}.** {section}")

st.markdown("---")
st.caption("💡 " + (tr("select_language") if lang == "es" else tr("select_language")))
