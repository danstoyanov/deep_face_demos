# --- START OF FILE app/main.py ---
# --- За да ръннеш приложението използвай тази команда в bash терминала ---
# -------------------------------------------------------------------------
#    ТЕСТВАНЕ И РЪН НА ПРОЕКТА 👇
# ⚠️      streamlit run app/main.py      ⚠️
#
# -------------------------------------------------------------------------

import streamlit as st

# Импорт на модулите за всеки режим
import face_analysis_image
import face_comparison
import video_emotion_analysis
import webcam_realtime_analysis
import language_model_chat
# <<< НОВ ИМПОРТ ЗА INSIGHTFACE АНАЛИЗ НА ИЗОБРАЖЕНИЕ
import face_analysis_image_insightface

# Импорт на помощни функции и конфигурация
import utils
import config

# --- Конфигурация на страницата ---
st.set_page_config(
    page_title="Мултифункционален Анализатор и Чат", layout="wide")
st.title("Мултифункционален Анализатор и Чатбот")
st.markdown("""
    Това приложение обединява няколко функционалности за анализ на лица и чат с езиков модел:
    **Анализ на изображение (DeepFace):** Разпознаване на възраст, пол и емоция.
    **Сравняване на две лица (InsightFace):** Качване на две изображения и изчисляване на сходството.
    **Анализ на емоции във видео (DeepFace):** Качване на видео и анализ на емоциите кадър по кадър.
    **Анализ от уеб камера (Real-time):** Разпознаване на възраст, пол и емоция в реално време.
    **Чат с Езиков Модел (Phi-3 Mini):** Интерактивен чат с локално стартиран LLM.
""")  # Описанието остава непроменено съгласно изискването да не се пипат други файлове освен main и новия.

# --- Зареждане на моделите за анализ на лица ---
# DeepFace модели
if 'deepface_models_loaded_status' not in st.session_state:
    st.session_state.deepface_models_loaded_status = utils.load_deepface_models()

# InsightFace модел
if 'insightface_model_app_object' not in st.session_state:
    if utils.INSIGHTFACE_AVAILABLE:
        st.session_state.insightface_model_app_object = utils.load_insightface_model()
    else:
        st.session_state.insightface_model_app_object = None
        # Предупреждението за липсващи InsightFace библиотеки ще се покаже от utils или при опит за използване
        # st.warning("Библиотеките 'insightface' или 'scikit-learn' не са намерени...")

# LLM моделът се зарежда и кешира от неговия собствен модул `language_model_chat.py` при нужда.

# --- Sidebar ---
st.sidebar.title("⚙️ Настройки и Режими")

analysis_modes_map = {
    "Анализ на изображение (DeepFace)": face_analysis_image,
    # <<< НОВ РЕЖИМ
    "Анализ на изображение (InsightFace)": face_analysis_image_insightface,
    "Сравняване на две лица (InsightFace)": face_comparison,
    "Анализ на емоции във видео (DeepFace)": video_emotion_analysis,
    "Анализ от уеб камера (Real-time)": webcam_realtime_analysis,
    "Чат с Езиков Модел (Phi-3)": language_model_chat
}
analysis_mode_name = st.sidebar.selectbox(
    "Изберете режим:",
    list(analysis_modes_map.keys())
)

# --- Извикване на избрания модул ---
selected_module = analysis_modes_map[analysis_mode_name]

if analysis_mode_name == "Анализ на изображение (DeepFace)":
    selected_module.render_page(
        deepface_models_loaded=st.session_state.deepface_models_loaded_status,
        MAX_DISPLAY_DIM=config.MAX_DISPLAY_DIM
    )
# <<< НОВ ELIF БЛОК
elif analysis_mode_name == "Анализ на изображение (InsightFace)":
    selected_module.render_page(
        insightface_available_flag=utils.INSIGHTFACE_AVAILABLE,
        insightface_model_app=st.session_state.get(
            'insightface_model_app_object'),
        max_display_dim_passed=config.MAX_DISPLAY_DIM
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
    st.sidebar.subheader("Настройки за видео анализ:")
    frame_skip_file = st.sidebar.slider(
        "Пропускай кадри за анализ (файл):", 0, 10, 1, key="video_module_frame_skip_file")
    detector_backend_video_file = st.sidebar.selectbox(
        "DeepFace детектор за видео (файл):",
        ('opencv', 'ssd', 'mtcnn', 'retinaface', 'yunet'), index=0, key="video_module_detector_file")
    selected_module.render_page(
        deepface_models_loaded=st.session_state.deepface_models_loaded_status
    )
elif analysis_mode_name == "Анализ от уеб камера (Real-time)":
    st.sidebar.subheader("Настройки за уеб камера:")
    detector_backend_webcam = st.sidebar.selectbox(
        "DeepFace детектор за уеб камера:",
        ('opencv', 'ssd', 'mtcnn', 'retinaface', 'yunet'), index=0, key="webcam_module_detector")
    selected_module.render_page(
        deepface_models_loaded=st.session_state.deepface_models_loaded_status
    )
elif analysis_mode_name == "Чат с Езиков Модел (Phi-3)":
    selected_module.render_page()


# --- Информация и лого в Sidebar (в края) ---
st.sidebar.markdown("---")
st.sidebar.info(
    "Приложение, базирано на [DeepFace](https://github.com/serengil/deepface), [InsightFace](https://github.com/deepinsight/insightface), [Llama CPP](https://github.com/abetlen/llama-cpp-python) и [Streamlit](https://streamlit.io/).")

utils.display_logo_in_sidebar(
    config.LOGO_PATH, config.LOGO_DISPLAY_WIDTH, config.LOGO_DISPLAY_WIDTH * 2)
