# --- START OF FILE second_streamlit_deepface_app.py ---
#
# --- За да ръннеш приложението използвай тази команда в bash терминала ---
# 🔴 streamlit run second_streamlit_deepface_app.py
# -------------------------------------------------------------------------

import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
# PIL is imported but not explicitly used for image processing here, which is fine as cv2 handles it.
from PIL import Image
import io
import os
import tempfile
import time
# import base64 # Може да не е нужен, ако не вграждаме видео като base64

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

# --- Общи помощни функции ---
MAX_DISPLAY_DIM = 600


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
# Моделите се зареждат само веднъж при стартиране на приложението или първото им използване в сесията,
# благодарение на st.cache_resource.


@st.cache_resource
def load_deepface_models():
    """Зарежда DeepFace моделите и ги кешира."""
    st.info("Зареждане/сваляне на DeepFace модели (може да отнеме време)...")
    try:
        # Изпълняваме фиктивен анализ, за да накараме DeepFace да зареди моделите
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
        DeepFace.analyze(dummy_image, actions=[
                         'age', 'gender', 'emotion'], enforce_detection=False, silent=True)
        st.success("DeepFace моделите са готови.")
        return True
    except Exception as e:
        st.error(f"Грешка при зареждане на DeepFace модели: {e}")
        return False  # Връщаме False, ако моделите не са заредени успешно.


deepface_models_loaded = load_deepface_models()

try:
    import insightface
    from insightface.app import FaceAnalysis
    from sklearn.metrics.pairwise import cosine_similarity
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    st.warning("Библиотеките 'insightface' или 'scikit-learn' не са намерени. Функцията за сравнение на лица няма да е достъпна. Моля, инсталирайте ги: `pip install insightface onnxruntime scikit-learn`")


@st.cache_resource
def load_insightface_model():
    """Зарежда InsightFace модела и го кешира."""
    if not INSIGHTFACE_AVAILABLE:
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
            app_insight.prepare(ctx_id=-1, det_size=(640, 640))
            st.success("InsightFace модел 'buffalo_l' е зареден с CPU.")
            return app_insight
        except Exception as e_cpu:
            st.error(
                f"Грешка при зареждане на InsightFace модел с CPU: {e_cpu}")
            return None


# Load InsightFace model only if available
insightface_model_app = None
if INSIGHTFACE_AVAILABLE:
    insightface_model_app = load_insightface_model()


st.sidebar.title("⚙️ Настройки и Режими")

# --- Избор на режим на анализ (остава тук, горе в sidebar) ---
analysis_mode = st.sidebar.selectbox(
    "Изберете режим на анализ:",
    ("Анализ на изображение (DeepFace)",
     "Сравняване на две лица (InsightFace)",
     "Анализ на емоции във видео (DeepFace)",
     "Анализ от уеб камера (Real-time)")
)

# ==============================================================================
# РЕЖИМ 1: АНАЛИЗ НА ИЗОБРАЖЕНИЕ (ВЪЗРАСТ, ПОЛ, ЕМОЦИЯ С DEEPFACE)
# ==============================================================================
if analysis_mode == "Анализ на изображение (DeepFace)":
    st.header("1. Анализ на изображение (Възраст, Пол, Емоция)")
    st.write("Качете изображение, за да анализирате лицата за възраст, емоция и пол, използвайки DeepFace.")

    if not deepface_models_loaded:
        st.error("DeepFace моделите не са заредени. Функционалността не е достъпна.")
        st.stop()

    uploaded_img_file = st.file_uploader(
        "Изберете изображение...", type=["jpg", "jpeg", "png"], key="deepface_image_uploader"
    )

    if uploaded_img_file is not None:
        file_bytes = np.asarray(
            bytearray(uploaded_img_file.read()), dtype=np.uint8)
        img_cv2_original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if img_cv2_original_bgr is None:
            st.error(
                "Не можах да прочета изображението. Моля, опитайте с друг файл.")
        else:
            st.subheader("Оригинално изображение:")
            img_display_resized_bgr = resize_image_for_display(
                img_cv2_original_bgr, MAX_DISPLAY_DIM)
            img_display_resized_rgb = cv2.cvtColor(
                img_display_resized_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_display_resized_rgb,
                     caption="Оригинално изображение (оразмерено)", use_column_width='auto')

            if st.button("🚀 Анализирай изображението (DeepFace)"):
                with st.spinner("Анализиране на лицата с DeepFace..."):
                    try:
                        face_analysis_results = DeepFace.analyze(
                            img_path=img_cv2_original_bgr.copy(),
                            actions=['age', 'gender', 'emotion'],
                            # False, за да не гърми, ако няма лице, но пак дава резултат.
                            enforce_detection=False,
                            # Може да стане избираем, но opencv е бърз и добър.
                            detector_backend='opencv',
                            silent=True
                        )
                    except Exception as e:
                        st.error(f"Възникна грешка при DeepFace анализ: {e}")
                        face_analysis_results = []

                img_cv2_annotated_bgr = img_cv2_original_bgr.copy()

                if not face_analysis_results or not isinstance(face_analysis_results, list):
                    st.info("Не бяха открити лица или резултатът е невалиден.")
                else:
                    num_faces_detected = 0
                    for i, face_result in enumerate(face_analysis_results):
                        # Проверка дали 'region' съществува и дали W/H са валидни (т.е. лице е детектнато)
                        if 'region' not in face_result or face_result['region']['w'] == 0 or face_result['region']['h'] == 0:
                            # Ако DeepFace върне 1 запис без регион, значи няма лице.
                            if len(face_analysis_results) == 1:
                                st.info("Не бяха открити лица в изображението.")
                            continue  # Пропускаме текущия невалиден резултат

                        num_faces_detected += 1
                        st.write(f"---")
                        st.write(f"#### Лице {num_faces_detected}:")

                        age = face_result.get('age', 'N/A')

                        dominant_gender_map = face_result.get('gender', {})
                        if dominant_gender_map:
                            dominant_gender = max(
                                dominant_gender_map, key=dominant_gender_map.get)
                            gender_confidence = dominant_gender_map[dominant_gender]
                        # Fallback, ако 'gender' не е dict (напр. само dominant_gender е върнато)
                        else:
                            dominant_gender = face_result.get(
                                'dominant_gender', 'N/A')
                            gender_confidence = 0  # Cannot determine confidence without the map

                        dominant_emotion_map = face_result.get('emotion', {})
                        if dominant_emotion_map:
                            dominant_emotion = max(
                                dominant_emotion_map, key=dominant_emotion_map.get)
                            emotion_confidence = dominant_emotion_map[dominant_emotion]
                        else:  # Fallback, ако 'emotion' не е dict
                            dominant_emotion = face_result.get(
                                'dominant_emotion', 'N/A')
                            emotion_confidence = 0  # Cannot determine confidence without the map

                        region = face_result['region']

                        st.write(f"- **Възраст:** {age}")
                        st.write(
                            f"- **Пол:** {dominant_gender} ({gender_confidence:.2f}%)")
                        st.write(
                            f"- **Емоция:** {dominant_emotion} ({emotion_confidence:.2f}%)")

                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        cv2.rectangle(img_cv2_annotated_bgr,
                                      (x, y), (x+w, y+h), (0, 255, 0), 2)
                        text_gender_char = dominant_gender[0].upper(
                            # Ensure gender is not N/A
                        ) if dominant_gender and isinstance(dominant_gender, str) and dominant_gender != "N/A" else "N"
                        text = f"A:{age} G:{text_gender_char} E:{dominant_emotion}"
                        cv2.putText(img_cv2_annotated_bgr, text, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    if num_faces_detected > 0:
                        st.subheader("Анотирано изображение:")
                        annotated_display_resized_bgr = resize_image_for_display(
                            img_cv2_annotated_bgr, MAX_DISPLAY_DIM)
                        annotated_display_resized_rgb = cv2.cvtColor(
                            annotated_display_resized_bgr, cv2.COLOR_BGR2RGB)
                        st.image(annotated_display_resized_rgb,
                                 caption=f"Анализирани лица: {num_faces_detected}", use_column_width='auto')
                    elif num_faces_detected == 0:
                        st.info(
                            "Не бяха открити лица в изображението, които да бъдат анализирани.")

# ==============================================================================
# РЕЖИМ 2: СРАВНЯВАНЕ НА ДВЕ ЛИЦА (INSIGHTFACE)
# ==============================================================================
elif analysis_mode == "Сравняване на две лица (InsightFace)":
    st.header("2. Сравняване на две лица (Cosine Similarity с InsightFace)")
    if not INSIGHTFACE_AVAILABLE or insightface_model_app is None:
        st.error(
            "InsightFace моделът не е зареден или библиотеката липсва. Функционалността не е достъпна.")
        st.stop()  # Спираме изпълнението, ако моделът не е наличен.

    col1, col2 = st.columns(2)
    with col1:
        uploaded_img1_file = st.file_uploader(
            "Изображение 1:", type=["jpg", "jpeg", "png"], key="insightface_img1")
    with col2:
        uploaded_img2_file = st.file_uploader(
            "Изображение 2:", type=["jpg", "jpeg", "png"], key="insightface_img2")

    def get_face_embedding_and_draw_insight(image_file, img_identifier_str, insight_app_model):
        if image_file is None:
            return None, None, f"Не е качен файл за Изображение {img_identifier_str}"

        file_bytes_img = np.asarray(
            bytearray(image_file.read()), dtype=np.uint8)
        img_cv_original_bgr = cv2.imdecode(file_bytes_img, cv2.IMREAD_COLOR)

        if img_cv_original_bgr is None:
            st.warning(
                f"Неуспешно зареждане на Изображение {img_identifier_str}.")
            return None, None, image_file.name

        faces = insight_app_model.get(img_cv_original_bgr)
        img_cv_annotated_bgr = img_cv_original_bgr.copy()

        if not faces:
            st.info(
                f"Не са намерени лица на Изображение {img_identifier_str} ({image_file.name}).")
            return None, img_cv_annotated_bgr, image_file.name

        main_face = faces[0]
        if len(faces) > 1:
            st.caption(
                f"Намерени {len(faces)} лица в Изображение {img_identifier_str}, използва се първото.")

        bbox = main_face.bbox.astype(int)
        cv2.rectangle(img_cv_annotated_bgr,
                      (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        embedding = main_face.embedding
        return embedding, img_cv_annotated_bgr, image_file.name

    if uploaded_img1_file and uploaded_img2_file:
        if st.button("🚀 Сравни лицата (InsightFace)"):
            with st.spinner("Обработка на изображенията и изчисляване на сходство..."):
                embedding1, img1_processed_bgr, filename1 = get_face_embedding_and_draw_insight(
                    uploaded_img1_file, "1", insightface_model_app)
                embedding2, img2_processed_bgr, filename2 = get_face_embedding_and_draw_insight(
                    uploaded_img2_file, "2", insightface_model_app)

                col_disp1, col_disp2 = st.columns(2)
                if img1_processed_bgr is not None:
                    with col_disp1:
                        resized_img1_bgr = resize_image_for_display(
                            img1_processed_bgr, MAX_DISPLAY_DIM // 2 + 100)
                        resized_img1_rgb = cv2.cvtColor(
                            resized_img1_bgr, cv2.COLOR_BGR2RGB)
                        st.image(resized_img1_rgb,
                                 caption=f"Изображение 1: {filename1}", use_column_width='auto')
                else:
                    with col_disp1:
                        st.warning(
                            f"Проблем с обработката на Изображение 1: {filename1}")

                if img2_processed_bgr is not None:
                    with col_disp2:
                        resized_img2_bgr = resize_image_for_display(
                            img2_processed_bgr, MAX_DISPLAY_DIM // 2 + 100)
                        resized_img2_rgb = cv2.cvtColor(
                            resized_img2_bgr, cv2.COLOR_BGR2RGB)
                        st.image(resized_img2_rgb,
                                 caption=f"Изображение 2: {filename2}", use_column_width='auto')
                else:
                    with col_disp2:
                        st.warning(
                            f"Проблем с обработката на Изображение 2: {filename2}")

                if embedding1 is not None and embedding2 is not None:
                    similarity_score = cosine_similarity(
                        embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
                    st.subheader(f"Резултат от сравнението:")
                    st.metric(label="Cosine Similarity",
                              value=f"{similarity_score:.4f}")
                    threshold = 0.58
                    if similarity_score >= threshold:
                        st.success(
                            f"Лицата са ВЕРОЯТНО на един и същи човек (сходство >= {threshold}).")
                    else:
                        st.warning(
                            f"Лицата са ВЕРОЯТНО на различни хора (сходство < {threshold}).")
                    st.caption(
                        f"(Използван праг: {threshold} за модел 'buffalo_l'. Стойностите са между -1 и 1. По-висока = по-голямо сходство.)")
                elif (uploaded_img1_file and uploaded_img2_file) and (embedding1 is None or embedding2 is None):
                    st.error(
                        "Не можа да се извлече лице от едното или двете изображения. Сравнението е невъзможно.")


# ==============================================================================
# РЕЖИМ 3: АНАЛИЗ НА ЕМОЦИИ ВЪВ ВИДЕО (DEEPFACE)
# ==============================================================================
elif analysis_mode == "Анализ на емоции във видео (DeepFace)":
    st.header("3. Анализ на емоции във видео (DeepFace)")
    if not deepface_models_loaded:
        st.error("DeepFace моделите не са заредени. Функционалността не е достъпна.")
        st.stop()

    uploaded_video_file = st.file_uploader(
        "Изберете видео файл (.mp4, .avi, .mov, .mkv)...", type=["mp4", "avi", "mov", "mkv"], key="deepface_video_uploader"
    )

    frame_skip_file = st.sidebar.slider(
        "Пропускай кадри за анализ (файл):", 0, 10, 1, key="video_frame_skip_file")
    detector_backend_video_file = st.sidebar.selectbox(
        "DeepFace детектор за видео (файл):",
        ('opencv', 'ssd', 'mtcnn', 'retinaface', 'yunet'), index=0, key="video_detector_file")

    # Управление на временните файлове за входно и изходно видео
    # Изчистване на стари временни файлове, ако е качен нов видеофайл
    if uploaded_video_file is not None:
        if 'temp_video_path' not in st.session_state or st.session_state.get('last_uploaded_video_name') != uploaded_video_file.name:
            # Премахваме стария временен видео файл, ако съществува
            if 'temp_video_path' in st.session_state and os.path.exists(st.session_state.temp_video_path):
                try:
                    os.remove(st.session_state.temp_video_path)
                    del st.session_state.temp_video_path
                except Exception as e:
                    st.warning(
                        f"Не мога да изтрия стария входен видео файл: {e}")
            # Премахваме стария временен изходен видео файл, ако съществува
            if 'temp_output_video_path' in st.session_state and os.path.exists(st.session_state.temp_output_video_path):
                try:
                    os.remove(st.session_state.temp_output_video_path)
                    del st.session_state.temp_output_video_path
                except Exception as e:
                    st.warning(
                        f"Не мога да изтрия стария изходен видео файл: {e}")

            # Записваме новия качен файл във временен файл
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_video_file.name.split('.')[-1]}") as tfile:
                tfile.write(uploaded_video_file.read())
                st.session_state.temp_video_path = tfile.name
                st.session_state.last_uploaded_video_name = uploaded_video_file.name

        temp_video_path = st.session_state.temp_video_path

        st.subheader("Оригинално видео:")
        col_orig_video, _ = st.columns([2, 1])
        with col_orig_video:
            if os.path.exists(temp_video_path):
                st.video(temp_video_path)
            else:
                st.warning(
                    "Оригиналният видео файл не е намерен. Моля, качете го отново.")
        st.caption(f"Оригиналното видео се показва с автоматично оразмеряване.")

        if st.button("🚀 Анализирай видеото (DeepFace)", key="analyze_video_button"):
            if not os.path.exists(temp_video_path):
                st.error(
                    "Грешка: Входният видео файл не е намерен. Моля, качете го отново.")
                st.stop()

            base_name, ext_name = os.path.splitext(uploaded_video_file.name)
            output_video_filename_for_download = f"{base_name}_emotions_processed.mp4"

            # Check if an output video for *this* input file has already been generated
            if 'temp_output_video_path' in st.session_state and os.path.exists(st.session_state.temp_output_video_path):
                if st.session_state.get('last_processed_video_base_name') != base_name:
                    try:
                        os.remove(st.session_state.temp_output_video_path)
                        del st.session_state.temp_output_video_path  # Clean up
                    except Exception:
                        pass  # Ignore if removal fails, might be in use

            if 'temp_output_video_path' not in st.session_state or not os.path.exists(st.session_state.get('temp_output_video_path', '')):
                st.session_state.temp_output_video_path = tempfile.NamedTemporaryFile(
                    delete=False, suffix="_processed.mp4").name
                st.session_state.last_processed_video_base_name = base_name
            temp_output_video_path = st.session_state.temp_output_video_path

            cap = None  # Initialize cap to None
            out_writer = None  # Initialize out_writer to None
            try:
                cap = cv2.VideoCapture(temp_video_path)
                if not cap.isOpened():
                    st.error(
                        f"Грешка: Не може да се отвори видео файлът: {uploaded_video_file.name}")
                    st.stop()

                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0 or fps is None or fps > 200:
                    st.warning(
                        f"Невалидна FPS стойност ({fps}) от видеото. Задавам 25.0 FPS.")
                    fps = 25.0

                # Препоръчително за .mp4 изход
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')

                out_writer = cv2.VideoWriter(
                    temp_output_video_path, fourcc, fps, (frame_width, frame_height))

                if not out_writer.isOpened():
                    st.error(f"Грешка: Не може да се инициализира VideoWriter с кодек 'mp4v'. Проверете вашата OpenCV/FFmpeg инсталация и наличните кодеци (напр. gstreamer плъгини или ffmpeg с подходяща поддръжка).")
                    st.stop()

                st.info(
                    f"Обработка на видео: {uploaded_video_file.name} ({frame_width}x{frame_height} @ {fps:.2f} FPS) с детектор '{detector_backend_video_file}'. Кодек: {'MP4V'}")
                progress_bar = st.progress(0)
                status_text = st.empty()
                percent_complete_text = st.empty()

                total_frames_prop = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                total_frames = int(
                    total_frames_prop) if total_frames_prop > 0 else 0

                processed_frame_count_for_analysis = 0
                current_frame_num_read = 0
                emotions_summary = {}
                start_time = time.time()
                last_known_faces_data = []  # Съхранява данни за лицата от последния анализиран кадър

                while cap.isOpened():
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break  # Край на видеото или грешка

                    current_frame_num_read += 1
                    frame_to_write = frame_bgr.copy()

                    # Анализираме всеки (frame_skip_file + 1)-ти кадър
                    # Ако frame_skip_file = 0, анализира се всеки кадър (0+1=1)
                    # Ако frame_skip_file = 1, анализира се всеки втори кадър (1+1=2)
                    if current_frame_num_read % (frame_skip_file + 1) == 1:
                        processed_frame_count_for_analysis += 1
                        try:
                            # DeepFace очаква enforce_detection=True, ако искате да връща само засечените лица с региони.
                            results = DeepFace.analyze(frame_bgr.copy(),
                                                       actions=['emotion'],
                                                       # За да се гарантира, че ще се върне регион на лице, ако е открито.
                                                       enforce_detection=True,
                                                       detector_backend=detector_backend_video_file,
                                                       silent=True)

                            current_faces_data = []
                            if isinstance(results, list) and len(results) > 0:
                                for result in results:
                                    # Уверете се, че 'region' съществува и има валидни размери
                                    if 'region' in result and result['region']['w'] > 0 and result['region']['h'] > 0:
                                        x, y, w, h = result['region']['x'], result['region'][
                                            'y'], result['region']['w'], result['region']['h']
                                        dominant_emotion_map = result.get(
                                            'emotion', {})
                                        if dominant_emotion_map:
                                            emotion = max(
                                                dominant_emotion_map, key=dominant_emotion_map.get)
                                            confidence = dominant_emotion_map[emotion]
                                        else:
                                            emotion = result.get(
                                                'dominant_emotion', "N/A")
                                            confidence = 0.0  # Няма стойност за увереност, ако липсва картата
                                        emotions_summary[emotion] = emotions_summary.get(
                                            emotion, 0) + 1
                                        current_faces_data.append(
                                            {'box': (x, y, w, h), 'text': f"{emotion} ({confidence:.2f})"})
                            if current_faces_data:  # Ако лица са засечени в текущия анализиран кадър, актуализираме last_known_faces_data
                                last_known_faces_data = current_faces_data
                            else:  # Ако няма засечени лица в текущия анализиран кадър, изчистваме last_known_faces_data
                                last_known_faces_data = []
                        # DeepFace хвърля ValueError, ако enforce_detection=True и не е открито лице.
                        except ValueError:
                            last_known_faces_data = []  # Няма лица за чертане
                            pass  # Продължаваме мълчаливо
                        except Exception as deepface_e:  # Хващаме други грешки от DeepFace
                            st.warning(
                                f"DeepFace грешка при анализ на кадър {current_frame_num_read}: {deepface_e}")
                            last_known_faces_data = []  # Няма лица за чертане
                            pass  # Продължаваме мълчаливо

                    # Рисуваме кутии и текст въз основа на last_known_faces_data (дори и ако анализът е бил пропуснат за този кадър)
                    if last_known_faces_data:
                        for face_data in last_known_faces_data:
                            x, y, w, h = face_data['box']
                            text = face_data['text']
                            cv2.rectangle(frame_to_write, (x, y),
                                          (x+w, y+h), (0, 255, 0), 2)
                            cv2.putText(frame_to_write, text, (x, y-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    out_writer.write(frame_to_write)

                    if total_frames > 0:
                        progress_val = int(
                            (current_frame_num_read / total_frames) * 100)
                        progress_bar.progress(min(progress_val, 100))
                        percent_complete_text.text(
                            f"{min(progress_val, 100)}% завършено")
                    status_text.text(
                        f"Прочетен кадър: {current_frame_num_read} / {total_frames if total_frames > 0 else 'N/A'}. Анализирани: {processed_frame_count_for_analysis}")

                end_time = time.time()
                progress_bar.empty()
                status_text.empty()
                percent_complete_text.empty()

                st.success(
                    f"Видео обработката приключи за {end_time - start_time:.2f} секунди.")
                st.info(
                    f"Общо прочетени кадри: {current_frame_num_read}. Анализирани: {processed_frame_count_for_analysis}")

                if os.path.exists(temp_output_video_path) and os.path.getsize(temp_output_video_path) > 0:
                    st.subheader("Обработено видео:")
                    col_proc_video, _ = st.columns([2, 1])
                    with col_proc_video:
                        st.video(temp_output_video_path)

                    with open(temp_output_video_path, 'rb') as video_file_for_download:
                        video_bytes_for_download = video_file_for_download.read()

                    st.download_button(
                        label="📥 Свали обработеното видео",
                        data=video_bytes_for_download,
                        file_name=output_video_filename_for_download,
                        mime="video/mp4",
                        key="download_processed_video"
                    )
                else:
                    st.error(
                        "Неуспешно генериране на изходното видео или файлът е празен.")

                if emotions_summary:
                    st.subheader(
                        "Обобщение на засечените доминиращи емоции (в анализираните кадри):")
                    sorted_emotions = dict(
                        sorted(emotions_summary.items(), key=lambda item: item[1], reverse=True))
                    st.bar_chart(sorted_emotions)
                else:
                    st.info(
                        "Не са засечени емоции по време на обработката (или не са намерени лица).")

            except Exception as e_video:
                st.error(
                    f"Възникна неочаквана грешка по време на видео обработката: {e_video}")
                import traceback
                st.error(traceback.format_exc())
            finally:
                # Освобождаване на ресурсите на камерата и писача
                if cap is not None:
                    cap.release()
                if out_writer is not None:
                    out_writer.release()
                # Почистване на временния входен видео файл
                if 'temp_video_path' in st.session_state and os.path.exists(st.session_state.temp_video_path):
                    try:
                        os.remove(st.session_state.temp_video_path)
                        del st.session_state.temp_video_path
                        if 'last_uploaded_video_name' in st.session_state:
                            del st.session_state.last_uploaded_video_name
                    except Exception as e:
                        st.warning(
                            f"Не мога да изтрия временния входен видео файл: {e}")
                # Временният изходен видео файл не се изтрива автоматично тук,
                # за да може потребителят да го изтегли.

# ==============================================================================
# РЕЖИМ 4: АНАЛИЗ ОТ УЕБ КАМЕРА (REAL-TIME)
# ==============================================================================
elif analysis_mode == "Анализ от уеб камера (Real-time)":
    st.header("4. Анализ от уеб камера (Пол, Възраст, Емоция - Real-time)")
    st.write("Стартирайте уеб камерата за анализ на лица в реално време.")
    st.warning(
        "Анализът в реално време може да е бавен в зависимост от вашия хардуер.")

    if not deepface_models_loaded:
        st.error("DeepFace моделите не са заредени. Функционалността не е достъпна.")
        st.stop()

    detector_backend_webcam = st.sidebar.selectbox(
        "DeepFace детектор за уеб камера:",
        ('opencv', 'ssd', 'mtcnn', 'retinaface', 'yunet'), index=0, key="webcam_detector")

    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False
    if 'cap' not in st.session_state:
        st.session_state.cap = None

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("🚀 Старт на камерата", key="start_cam_button", disabled=st.session_state.webcam_running):
            st.session_state.webcam_running = True
            st.rerun()

    with col_stop:
        if st.button("🛑 Стоп на камерата", key="stop_cam_button", disabled=not st.session_state.webcam_running):
            st.session_state.webcam_running = False
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.rerun()

    image_placeholder = st.empty()

    if st.session_state.webcam_running:
        if st.session_state.cap is None:
            st.session_state.cap = cv2.VideoCapture(
                0)  # 0 за стандартна уеб камера
            if not st.session_state.cap.isOpened():
                st.error(
                    "Не може да се отвори уеб камерата. Проверете дали е свързана и не се използва от друго приложение.")
                st.session_state.webcam_running = False
                st.session_state.cap = None

        if st.session_state.cap is not None and st.session_state.cap.isOpened():
            st.info("Уеб камерата е активна...")
            while st.session_state.webcam_running:
                ret, frame_bgr = st.session_state.cap.read()
                if not ret:
                    st.warning(
                        "Не може да се прочете кадър от уеб камерата. Спирам.")
                    st.session_state.webcam_running = False
                    break

                frame_to_analyze_bgr = frame_bgr.copy()
                frame_display_bgr = frame_bgr.copy()

                try:
                    # Enforce detection in webcam to ensure a face region is returned for drawing
                    results = DeepFace.analyze(
                        img_path=frame_to_analyze_bgr,
                        actions=['age', 'gender', 'emotion'],
                        enforce_detection=True,  # За да се гарантира, че ще се върне регион на лице
                        detector_backend=detector_backend_webcam,
                        silent=True
                    )

                    if isinstance(results, list) and len(results) > 0:
                        for face_info in results:
                            # Обработваме само ако е засечен валиден регион на лицето
                            if 'region' in face_info and face_info['region']['w'] > 0 and face_info['region']['h'] > 0:
                                region = face_info['region']
                                x, y, w, h = region['x'], region['y'], region['w'], region['h']

                                age = face_info.get('age', "N/A")

                                gender_map = face_info.get('gender', {})
                                dominant_gender = max(gender_map, key=gender_map.get) if gender_map else face_info.get(
                                    'dominant_gender', "N/A")

                                emotion_map = face_info.get('emotion', {})
                                dominant_emotion = max(emotion_map, key=emotion_map.get) if emotion_map else face_info.get(
                                    'dominant_emotion', "N/A")

                                cv2.rectangle(frame_display_bgr, (x, y),
                                              (x + w, y + h), (0, 255, 0), 2)
                                text_gender_char = dominant_gender[0].upper(
                                ) if dominant_gender != "N/A" and isinstance(dominant_gender, str) else "N"
                                text = f"A:{age} G:{text_gender_char} E:{dominant_emotion}"
                                cv2.putText(frame_display_bgr, text, (x, y - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            elif len(results) == 1 and face_info['region']['w'] == 0:
                                # Това е случаят, когато DeepFace връща един резултат, който не е открил лице
                                pass
                except Exception as e:
                    # Хващаме всякакви грешки при анализ (напр. ValueError, ако не е открито лице при enforce_detection=True)
                    pass

                frame_display_rgb = cv2.cvtColor(
                    frame_display_bgr, cv2.COLOR_BGR2RGB)
                image_placeholder.image(frame_display_rgb, channels="RGB")

                # Малко забавяне, за да намали натоварването на процесора и да позволи на Streamlit да актуализира UI
                time.sleep(0.01)

                if not st.session_state.webcam_running:
                    break  # Излизаме от цикъла, ако бутонът за спиране е натиснат

            # Освобождаваме камерата и изчистваме плейсхолдъра след края на цикъла
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            image_placeholder.empty()
            if not st.session_state.webcam_running:
                st.info("Уеб камерата е спряна.")

        elif st.session_state.cap is None and st.session_state.webcam_running:
            # Този случай означава, че webcam_running е True, но cap е None (неуспешно отваряне)
            st.session_state.webcam_running = False  # Нулираме състоянието
            st.error("Грешка: Камерата не можа да бъде стартирана.")


# --- Преместени елементи в края на страничната лента ---
st.sidebar.markdown("---")
st.sidebar.info(
    "Приложение, базирано на [DeepFace](https://github.com/serengil/deepface), [InsightFace](https://github.com/deepinsight/insightface) и [Streamlit](https://streamlit.io/).")

# --- Добавяне на логото в страничната лента (sidebar) ---
LOGO_PATH = "bdu_black_logo.jpg"  # ПРОМЯНА: Променено име на файла
LOGO_DISPLAY_WIDTH = 100  # Максимална ширина за логото в пиксели за по-малък размер

st.sidebar.markdown("---")  # Разделител преди логото
if os.path.exists(LOGO_PATH):
    try:
        logo_bgr = cv2.imread(LOGO_PATH)
        if logo_bgr is not None:
            # Използваме resize_image_for_display, за да гарантираме, че NumPy масивът е с разумен размер
            # преди да го подадем на Streamlit, и след това изрично задаваме width за показване.
            resized_logo_bgr = resize_image_for_display(
                logo_bgr, max_dim=LOGO_DISPLAY_WIDTH)
            resized_logo_rgb = cv2.cvtColor(
                resized_logo_bgr, cv2.COLOR_BGR2RGB)
            st.sidebar.image(
                resized_logo_rgb, width=LOGO_DISPLAY_WIDTH)  # Премахнат caption тук
        else:
            st.sidebar.warning(
                f"Не мога да заредя логото от '{LOGO_PATH}'. Проверете пътя или формата на файла.")
    except Exception as e:
        st.sidebar.error(f"Грешка при показване на логото: {e}")
else:
    st.sidebar.info(
        f"За да добавите лого, поставете файл на име '{LOGO_PATH}' (или променете пътя) в същата директория като приложението.")

# Текстът "Създадено от екипа на БДУ." преместен тук
st.sidebar.markdown("Разработено от КСТ – Бургаски държавен университет (БДУ)")
st.sidebar.markdown("---")  # Разделител след логото (по желание)
# --- Край на добавянето на лого и текст ---


# --- END OF FILE second_streamlit_deepface_app.py ---
