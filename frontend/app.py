import streamlit as st
import os
from utils import api_client

st.set_page_config(
    page_title="California Housing ML",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modern CSS ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0e0;
    }
    .stMarkdown, .stText, p, li, h1, h2, h3, h4, h5, h6 { color: #e0e0e0 !important; }
    h1 { font-weight: 700 !important; background: linear-gradient(90deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    h2, h3 { font-weight: 600 !important; color: #c0c0ff !important; }
    .stButton>button {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
        color: white !important; border: none !important; border-radius: 8px !important;
        font-weight: 500 !important; padding: 0.5rem 1.5rem !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px); box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background: rgba(255,255,255,0.08) !important; border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 8px !important; color: #e0e0e0 !important;
    }
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border-color: #667eea !important; box-shadow: 0 0 0 2px rgba(102,126,234,0.3) !important;
    }
    .stSelectbox>div>div>div { background: rgba(255,255,255,0.08) !important; border-radius: 8px !important; color: #e0e0e0 !important; }
    .stSlider>div>div>div>div { background: #667eea !important; }
    .st-bd { border-color: rgba(255,255,255,0.1) !important; }
    .stDataFrame { background: rgba(255,255,255,0.05) !important; border-radius: 10px !important; overflow: hidden; }
    .stDataFrame th { background: rgba(102,126,234,0.2) !important; color: #c0c0ff !important; font-weight: 600 !important; }
    .stDataFrame td { color: #e0e0e0 !important; }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.06) !important; border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px !important; padding: 16px !important; text-align: center;
        backdrop-filter: blur(10px);
    }
    div[data-testid="stMetric"] label { color: #a0a0c0 !important; font-weight: 500 !important; }
    div[data-testid="stMetric"] div { color: #e0e0ff !important; font-size: 1.8rem !important; font-weight: 700 !important; }
    .stSidebar { background: rgba(15,12,41,0.95) !important; border-right: 1px solid rgba(255,255,255,0.08); }
    .stSidebar .stMarkdown, .stSidebar p, .stSidebar h1, .stSidebar h2, .stSidebar h3 { color: #c0c0ff !important; }
    .stSidebar .stRadio>div>label>div { color: #e0e0e0 !important; font-weight: 400; }
    .stSidebar .stRadio>div>label>div:hover { color: #667eea !important; }
    .st-bw { background: transparent !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: rgba(255,255,255,0.05); border-radius: 12px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px !important; padding: 8px 20px !important; color: #a0a0c0 !important; }
    .stTabs [aria-selected="true"] { background: linear-gradient(90deg, #667eea, #764ba2) !important; color: white !important; }
    .st-expander { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 12px !important; }
    .st-info, .stAlert { background: rgba(102,126,234,0.15) !important; border: 1px solid rgba(102,126,234,0.3) !important; color: #c0c0ff !important; border-radius: 10px !important; }
    .st-bq { border-left-color: #667eea !important; }
    hr { border-color: rgba(255,255,255,0.08) !important; }
    section[data-testid="stSidebar"] { width: 280px !important; }
    .login-container { max-width: 420px; margin: 0 auto; padding: 2rem; }
    .card {
        background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px; padding: 24px; backdrop-filter: blur(12px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(102,126,234,0.2); }
    .stPlotlyChart { background: rgba(255,255,255,0.03); border-radius: 12px; padding: 8px; }
</style>
""", unsafe_allow_html=True)

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
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<div class='login-container'>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align:center;font-size:2.2rem;'>{tr('app_title')}</h1>", unsafe_allow_html=True)
        st.markdown("<div class='card' style='margin-top:1.5rem;'>", unsafe_allow_html=True)
        
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
        
        st.markdown("</div></div>", unsafe_allow_html=True)

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
