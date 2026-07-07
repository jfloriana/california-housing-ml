import streamlit as st
import os
from utils import api_client

st.set_page_config(
    page_title="California Housing ML",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed",
)



# Session state initialization
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = None
if "language" not in st.session_state:
    st.session_state.language = "es"
if "page" not in st.session_state:
    st.session_state.page = "Login"
if "X_data" not in st.session_state:
    st.session_state.X_data = None
if "y_data" not in st.session_state:
    st.session_state.y_data = None
if "feature_names" not in st.session_state:
    st.session_state.feature_names = None
if "trained_results" not in st.session_state:
    st.session_state.trained_results = None
if "predictions_store" not in st.session_state:
    st.session_state.predictions_store = None
if "y_test_data" not in st.session_state:
    st.session_state.y_test_data = None
if "api_metrics_cache" not in st.session_state:
    st.session_state.api_metrics_cache = {}

# Translation dictionary
LANG = {
    "es": {
        "app_title": "California Housing - Regresión con Redes Neuronales",
        "login": "Iniciar Sesión",
        "signup": "Registrarse",
        "email": "Correo electrónico",
        "password": "Contraseña",
        "login_btn": "Ingresar",
        "signup_btn": "Registrarse",
        "logout": "Cerrar Sesión",
        "welcome": "Bienvenido",
        "error_auth": "Error de autenticación",
        # Navigation
        "nav_eda": "EDA - Análisis Exploratorio",
        "nav_training": "Entrenamiento de Modelos",
        "nav_cv": "Validación Cruzada",
        "nav_tuning": "Hiperparámetros",
        "nav_stats": "Pruebas Estadísticas",
        "nav_reports": "Reportes",
        "nav_dashboard": "Dashboard General",
        "nav_predict": "Predecir",
        "language": "Idioma",
        "spanish": "Español",
        "english": "Inglés",
    },
    "en": {
        "app_title": "California Housing - Neural Network Regression",
        "login": "Login",
        "signup": "Sign Up",
        "email": "Email",
        "password": "Password",
        "login_btn": "Sign In",
        "signup_btn": "Sign Up",
        "logout": "Logout",
        "welcome": "Welcome",
        "error_auth": "Authentication Error",
        "nav_eda": "EDA - Exploratory Analysis",
        "nav_training": "Model Training",
        "nav_cv": "Cross Validation",
        "nav_tuning": "Hyperparameter Tuning",
        "nav_stats": "Statistical Tests",
        "nav_reports": "Reports",
        "nav_dashboard": "General Dashboard",
        "nav_predict": "Predict",
        "language": "Language",
        "spanish": "Spanish",
        "english": "English",
    },
}

def tr(key):
    return LANG.get(st.session_state.language, LANG["es"]).get(key, key)

# ── Login Page ──────────────────────────────────────────────────────
def login_page():
    st.markdown("""
<style>
    section[data-testid="stSidebar"] { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }
</style>
""", unsafe_allow_html=True)
    st.title(tr("app_title"))
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs([tr("login"), tr("signup")])
        
        with tab1:
            with st.form("login_form"):
                email = st.text_input(tr("email"), value="test@test.com")
                password = st.text_input(tr("password"), type="password", value="test123")
                if st.form_submit_button(tr("login_btn"), use_container_width=True):
                    try:
                        result = api_client.signin(email, password)
                        st.session_state.authenticated = True
                        st.session_state.token = result["access_token"]
                        st.session_state.user = result["user"]
                        try:
                            prefs = api_client.get_preferences(result["access_token"])
                            st.session_state.language = prefs.get("language", "es")
                        except:
                            pass
                        st.rerun()
                    except Exception as e:
                        st.error(f"{tr('error_auth')}: {str(e)}")
        
        with tab2:
            with st.form("signup_form"):
                new_email = st.text_input(tr("email"))
                new_password = st.text_input(tr("password"), type="password")
                if st.form_submit_button(tr("signup_btn"), use_container_width=True):
                    try:
                        result = api_client.signup(new_email, new_password)
                        st.success("User created! Check your email for confirmation.")
                    except Exception as e:
                        st.error(f"{tr('error_auth')}: {str(e)}")

# ── Main App (authenticated) ───────────────────────────────────────
def main_app():
    st.sidebar.title(tr("app_title"))
    st.sidebar.markdown(f"**{tr('welcome')}**, {st.session_state.user.get('email', '')}")
    
    # Language selector
    lang_label = tr("language")
    lang_options = [tr("spanish"), tr("english")]
    lang_values = ["es", "en"]
    current_idx = lang_values.index(st.session_state.language) if st.session_state.language in lang_values else 0
    
    selected_lang = st.sidebar.selectbox(lang_label, lang_options, index=current_idx)
    new_lang = lang_values[lang_options.index(selected_lang)]
    
    if new_lang != st.session_state.language:
        st.session_state.language = new_lang
        try:
            api_client.update_preferences(st.session_state.token, new_lang)
        except:
            pass
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Navigation
    nav_options = [
        tr("nav_dashboard"),
        tr("nav_eda"),
        tr("nav_training"),
        tr("nav_cv"),
        tr("nav_tuning"),
        tr("nav_stats"),
        tr("nav_reports"),
        tr("nav_predict"),
    ]
    
    selected_page = st.sidebar.radio("", nav_options, index=0)
    
    st.sidebar.markdown("---")
    if st.sidebar.button(tr("logout"), use_container_width=True):
        st.session_state.authenticated = False
        st.session_state.token = None
        st.session_state.user = None
        st.rerun()
    
    # Page routing
    page_map = {
        tr("nav_dashboard"): "pages/7_Dashboard.py",
        tr("nav_eda"): "pages/1_EDA.py",
        tr("nav_training"): "pages/2_Entrenamiento.py",
        tr("nav_cv"): "pages/3_Validacion_Cruzada.py",
        tr("nav_tuning"): "pages/4_Hiperparametros.py",
        tr("nav_stats"): "pages/5_Pruebas_Estadisticas.py",
        tr("nav_reports"): "pages/6_Reportes.py",
        tr("nav_predict"): "pages/8_Predict.py",
    }
    
    page_file = page_map.get(selected_page)
    if page_file:
        try:
            with open(page_file, "r", encoding="utf-8") as f:
                code = f.read()
            exec(code, globals())
        except FileNotFoundError:
            st.info(f"Page not found: {page_file}")
    else:
        st.info("Select a page from the sidebar")

# ── Routing ─────────────────────────────────────────────────────────
if st.session_state.authenticated:
    main_app()
else:
    login_page()
