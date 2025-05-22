# --- За да ръннеш приложението използвай тази команда в bash терминала ---
# 🔴 streamlit run app/main.py
# -------------------------------------------------------------------------

import streamlit as st

# Импорт на модулите за всеки режим
import face_analysis_image
import face_comparison
import video_emotion_analysis
import webcam_realtime_analysis

# Импорт на помощни функции и конфигурация
import utils
# Трябва да е app.config ако main.py е извън app папката, но тук е в нея.
import config

# --- Конфигурация на страницата ---
st.set_page_config(
    page_title="Мултифункционален Анализатор на Лица", layout="wide")
st.title("Мултифункционален Анализатор на Лица")
st.markdown("""
    Това приложение обединява няколко функционалности за анализ на лица:
    1.  **Анализ на изображение (DeepFace):** Разпознаване на възраст, пол и емоция.
    2.  **Сравняване на две лица (InsightFace):** Качване на две изображения и изчисляване на сходството между лицата.
    3.  **Анализ на емоции във видео (DeepFace):** Качване на видео и анализ на емоциите кадър по кадър.
    4.  **Анализ от уеб камера (Real-time):** Разпознаване на възраст, пол и емоция в реално време.
""")

# --- Зареждане на моделите ---
# DeepFace модели
if 'deepface_models_loaded_status' not in st.session_state:
    st.session_state.deepface_models_loaded_status = utils.load_deepface_models()

# InsightFace модел
if 'insightface_model_app_object' not in st.session_state:
    if utils.INSIGHTFACE_AVAILABLE:
        # Функцията utils.load_insightface_model() сама ще покаже съобщенията st.info/success/error
        st.session_state.insightface_model_app_object = utils.load_insightface_model()
    else:
        st.session_state.insightface_model_app_object = None
        # Показваме предупреждение в UI, ако библиотеките липсват
        st.warning("Библиотеките 'insightface' или 'scikit-learn' не са намерени. Функцията за сравнение на лица няма да е достъпна. Моля, инсталирайте ги: `pip install insightface onnxruntime scikit-learn`")

# --- Sidebar ---
st.sidebar.title("⚙️ Настройки и Режими")

analysis_modes_map = {
    "Анализ на изображение (DeepFace)": face_analysis_image,
    "Сравняване на две лица (InsightFace)": face_comparison,
    "Анализ на емоции във видео (DeepFace)": video_emotion_analysis,
    "Анализ от уеб камера (Real-time)": webcam_realtime_analysis
}
analysis_mode_name = st.sidebar.selectbox(
    "Изберете режим на анализ:",
    list(analysis_modes_map.keys())
)

# --- Извикване на избрания модул ---
selected_module = analysis_modes_map[analysis_mode_name]

if analysis_mode_name == "Анализ на изображение (DeepFace)":
    selected_module.render_page(
        deepface_models_loaded=st.session_state.deepface_models_loaded_status,
        MAX_DISPLAY_DIM=config.MAX_DISPLAY_DIM
    )
elif analysis_mode_name == "Сравняване на две лица (InsightFace)":
    selected_module.render_page(
        INSIGHTFACE_AVAILABLE=utils.INSIGHTFACE_AVAILABLE,
        insightface_model_app=st.session_state.get(
            'insightface_model_app_object'),
        MAX_DISPLAY_DIM=config.MAX_DISPLAY_DIM,
        INSIGHTFACE_THRESHOLD=config.INSIGHTFACE_THRESHOLD
    )
elif analysis_mode_name == "Анализ на емоции във видео (DeepFace)":
    selected_module.render_page(
        deepface_models_loaded=st.session_state.deepface_models_loaded_status
        # Конфигурационните константи като DEFAULT_VIDEO_FPS ще се импортират директно в модула от config.py
    )
elif analysis_mode_name == "Анализ от уеб камера (Real-time)":
    selected_module.render_page(
        deepface_models_loaded=st.session_state.deepface_models_loaded_status
    )

# --- Информация и лого в Sidebar ---
st.sidebar.markdown("---")
st.sidebar.info(
    "Приложение, базирано на [DeepFace](https://github.com/serengil/deepface), [InsightFace](https://github.com/deepinsight/insightface) и [Streamlit](https://streamlit.io/).")

# Показване на логото, подаваме и максималния размер за преоразмеряване на самото лого изображение
utils.display_logo_in_sidebar(config.LOGO_PATH, config.LOGO_DISPLAY_WIDTH,
                              config.LOGO_DISPLAY_WIDTH * 2)  # *2 за по-добро качество при resize

