# --- START OF FILE app/main.py ---
# --- За да ръннеш приложението използвай тази команда в bash терминала ---
# 🔴 streamlit run app/main.py
# -------------------------------------------------------------------------

import streamlit as st

# Импорт на модулите за всеки режим
import face_analysis_image
import face_comparison
import video_emotion_analysis
import webcam_realtime_analysis
import language_model_chat  # <<< НОВ ИМПОРТ

# Импорт на помощни функции и конфигурация
import utils
import config

# --- Конфигурация на страницата ---
st.set_page_config(
    page_title="Мултифункционален Анализатор и Чат", layout="wide")
st.title("Мултифункционален Анализатор и Чатбот")
st.markdown("""
    Това приложение обединява няколко функционалности за анализ на лица и чат с езиков модел:
    1.  **Анализ на изображение (DeepFace):** Разпознаване на възраст, пол и емоция.
    2.  **Сравняване на две лица (InsightFace):** Качване на две изображения и изчисляване на сходството.
    3.  **Анализ на емоции във видео (DeepFace):** Качване на видео и анализ на емоциите кадър по кадър.
    4.  **Анализ от уеб камера (Real-time):** Разпознаване на възраст, пол и емоция в реално време.
    5.  **Чат с Езиков Модел (Phi-3 Mini):** Интерактивен чат с локално стартиран LLM.
""")

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
    "Сравняване на две лица (InsightFace)": face_comparison,
    "Анализ на емоции във видео (DeepFace)": video_emotion_analysis,
    "Анализ от уеб камера (Real-time)": webcam_realtime_analysis,
    "Чат с Езиков Модел (Phi-3)": language_model_chat  # <<< НОВ РЕЖИМ
}
analysis_mode_name = st.sidebar.selectbox(
    "Изберете режим:",  # По-кратко име
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
        INSIGHTFACE_AVAILABLE=utils.INSIGHTFACE_AVAILABLE,  # Подаване на флага
        insightface_model_app=st.session_state.get(
            'insightface_model_app_object'),
        MAX_DISPLAY_DIM=config.MAX_DISPLAY_DIM,
        INSIGHTFACE_THRESHOLD=config.INSIGHTFACE_THRESHOLD
    )
elif analysis_mode_name == "Анализ на емоции във видео (DeepFace)":
    # Показване на специфичните за видео sidebar контроли само когато този режим е активен
    st.sidebar.subheader("Настройки за видео анализ:")
    frame_skip_file = st.sidebar.slider(  # Ключовете са променени да са уникални
        "Пропускай кадри за анализ (файл):", 0, 10, 1, key="video_module_frame_skip_file")
    detector_backend_video_file = st.sidebar.selectbox(
        "DeepFace детектор за видео (файл):",
        ('opencv', 'ssd', 'mtcnn', 'retinaface', 'yunet'), index=0, key="video_module_detector_file")

    # Предаваме тези стойности на render_page, ако е нужно, или модулът ги чете директно от st.sidebar
    # Засега модулът video_emotion_analysis.py ги чете директно от st.sidebar с уникални ключове.
    selected_module.render_page(
        deepface_models_loaded=st.session_state.deepface_models_loaded_status
        # frame_skip=frame_skip_file, # Пример ако искаме да ги подадем
        # detector_backend=detector_backend_video_file
    )
elif analysis_mode_name == "Анализ от уеб камера (Real-time)":
    # Показване на специфичните за уеб камера sidebar контроли
    st.sidebar.subheader("Настройки за уеб камера:")
    detector_backend_webcam = st.sidebar.selectbox(
        "DeepFace детектор за уеб камера:",
        ('opencv', 'ssd', 'mtcnn', 'retinaface', 'yunet'), index=0, key="webcam_module_detector")

    selected_module.render_page(
        deepface_models_loaded=st.session_state.deepface_models_loaded_status
        # detector_backend=detector_backend_webcam # Модулът webcam_realtime_analysis.py го чете директно
    )
elif analysis_mode_name == "Чат с Езиков Модел (Phi-3)":  # <<< НОВ ELIF БЛОК
    # За LLM чата може да няма специфични настройки в sidebar засега,
    # или могат да се добавят директно в неговия render_page или тук.
    selected_module.render_page()


# --- Информация и лого в Sidebar (в края) ---
st.sidebar.markdown("---")  # Разделител преди общата информация
st.sidebar.info(
    "Приложение, базирано на [DeepFace](https://github.com/serengil/deepface), [InsightFace](https://github.com/deepinsight/insightface), [Llama CPP](https://github.com/abetlen/llama-cpp-python) и [Streamlit](https://streamlit.io/).")

utils.display_logo_in_sidebar(
    config.LOGO_PATH, config.LOGO_DISPLAY_WIDTH, config.LOGO_DISPLAY_WIDTH * 2)
