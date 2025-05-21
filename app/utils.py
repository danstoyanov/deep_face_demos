# --- START OF FILE app/utils.py ---
import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
import os
# import time # Не се използва директно тук, но е налично ако е нужно

# PIL is imported in original but not directly used for image processing by these utils, cv2 handles it.
# import base64 # Not used

# --- Общи помощни функции ---


@st.cache_data
def resize_image_for_display(image_np, max_dim):
    """
    Преоразмерява изображение, за да се побере в максимални размери за дисплей,
    запазвайки пропорциите.
    """
    height, width = image_np.shape[:2]
    if width <= max_dim and height <= max_dim:
        return image_np
    scale = max_dim / max(width, height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(
        image_np, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized_image

# --- Кеширане на модели ---


@st.cache_resource
def load_deepface_models():
    """Зарежда DeepFace моделите и ги кешира."""
    st.info("Зареждане/сваляне на DeepFace модели (може да отнеме време)...")
    try:
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        DeepFace.analyze(dummy_image, actions=[
                         'age', 'gender', 'emotion'], enforce_detection=False, silent=True)
        st.success("DeepFace моделите са готови.")
        return True  # Връщаме True, ако моделите са заредени успешно.
    except Exception as e:
        st.error(f"Грешка при зареждане на DeepFace модели: {e}")
        return False  # Връщаме False, ако има грешка.


# Global state for InsightFace availability, checked once at import time.
INSIGHTFACE_AVAILABLE = False
try:
    import insightface
    from insightface.app import FaceAnalysis
    # sklearn.metrics.pairwise.cosine_similarity ще бъде импортиран в face_comparison.py при нужда
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    # Това съобщение ще се изпише в конзолата, ако библиотеките липсват.
    # Streamlit UI предупреждението ще се покаже в main.py.
    print("ПРЕДУПРЕЖДЕНИЕ: Библиотеките 'insightface' или 'scikit-learn' не са намерени. Функцията за сравнение на лица няма да е достъпна.")
    pass


@st.cache_resource
def load_insightface_model():
    """Зарежда InsightFace модела и го кешира. Показва UI съобщения."""
    if not INSIGHTFACE_AVAILABLE:
        # st.warning тук няма да се покаже, тъй като main.py вече ще е показал предупреждение.
        # Функцията просто няма да бъде извикана от main.py ако INSIGHTFACE_AVAILABLE е False.
        # Ако все пак бъде извикана, това е предпазна мярка.
        st.warning(
            "InsightFace библиотека не е достъпна, моделът няма да бъде зареден.")
        return None

    st.info("Зареждане/сваляне на InsightFace модел (buffalo_l)...")
    try:
        app_insight = FaceAnalysis(name='buffalo_l', providers=[
                                   'CUDAExecutionProvider', 'CPUExecutionProvider'])
        app_insight.prepare(ctx_id=0, det_size=(640, 640))
        st.success("InsightFace модел 'buffalo_l' е зареден (опитва GPU).")
        return app_insight
    except Exception as e_gpu:
        st.warning(
            f"Неуспешно зареждане на InsightFace с GPU ({e_gpu}). Опитвам само с CPU...")
        try:
            app_insight = FaceAnalysis(name='buffalo_l', providers=[
                                       'CPUExecutionProvider'])
            # ctx_id = -1 за CPU
            app_insight.prepare(ctx_id=-1, det_size=(640, 640))
            st.success("InsightFace модел 'buffalo_l' е зареден с CPU.")
            return app_insight
        except Exception as e_cpu:
            st.error(
                f"Грешка при зареждане на InsightFace модел с CPU: {e_cpu}")
            return None


def display_logo_in_sidebar(logo_path, display_width, max_logo_dim_for_resize):
    st.sidebar.markdown("---")
    if os.path.exists(logo_path):
        try:
            logo_bgr = cv2.imread(logo_path)
            if logo_bgr is not None:
                # Преоразмеряваме логото преди да го подадем на st.image, за да е сигурно, че е малък NumPy масив
                resized_logo_bgr = resize_image_for_display(
                    logo_bgr, max_dim=max_logo_dim_for_resize)
                resized_logo_rgb = cv2.cvtColor(
                    resized_logo_bgr, cv2.COLOR_BGR2RGB)
                st.sidebar.image(resized_logo_rgb, width=display_width)
            else:
                st.sidebar.warning(
                    f"Не мога да заредя логото от '{logo_path}'.")
        except Exception as e:
            st.sidebar.error(f"Грешка при показване на логото: {e}")
    else:
        st.sidebar.info(f"Лого файл '{logo_path}' не е намерен.")

    st.sidebar.markdown(
        "Разработено от КСТ – Бургаски държавен университет (БДУ)")
    st.sidebar.markdown("---")
# --- END OF FILE app/utils.py ---
