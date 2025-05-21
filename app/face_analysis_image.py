# --- START OF FILE app/face_analysis_image.py ---
import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from utils import resize_image_for_display  # Импорт от utils.py
# MAX_DISPLAY_DIM ще се подаде като аргумент от main.py


def render_page(deepface_models_loaded, MAX_DISPLAY_DIM):
    st.header("1. Анализ на изображение (Възраст, Пол, Емоция)")
    st.write("Качете изображение, за да анализирате лицата за възраст, емоция и пол, използвайки DeepFace.")

    if not deepface_models_loaded:
        st.error("DeepFace моделите не са заредени. Функционалността не е достъпна.")
        st.stop()

    uploaded_img_file = st.file_uploader(
        # Уникален ключ за file_uploader
        "Изберете изображение...", type=["jpg", "jpeg", "png"], key="deepface_image_uploader_module"
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

            if st.button("🚀 Анализирай изображението (DeepFace)", key="analyze_image_deepface_btn"):
                with st.spinner("Анализиране на лицата с DeepFace..."):
                    try:
                        face_analysis_results = DeepFace.analyze(
                            img_path=img_cv2_original_bgr.copy(),  # Изпращаме копие
                            actions=['age', 'gender', 'emotion'],
                            enforce_detection=False,
                            detector_backend='opencv',  # Може да стане избираемо
                            silent=True
                        )
                    except Exception as e:
                        st.error(f"Възникна грешка при DeepFace анализ: {e}")
                        face_analysis_results = []  # Празен списък при грешка

                img_cv2_annotated_bgr = img_cv2_original_bgr.copy()

                if not face_analysis_results or not isinstance(face_analysis_results, list):
                    # Този случай е ако DeepFace върне нещо неочаквано (не списък)
                    st.info("Анализът не върна валидни резултати.")
                else:
                    num_faces_detected = 0
                    for i, face_result in enumerate(face_analysis_results):
                        # Проверка дали 'region' съществува и дали W/H са валидни
                        # DeepFace с enforce_detection=False връща 1 запис с w=0, h=0 ако няма лице
                        if 'region' not in face_result or face_result['region']['w'] == 0 or face_result['region']['h'] == 0:
                            # Пропускаме, ако няма валиден регион (т.е. не е засечено лице)
                            continue

                        num_faces_detected += 1
                        st.write(f"---")
                        st.write(f"#### Лице {num_faces_detected}:")

                        age = face_result.get('age', 'N/A')

                        dominant_gender_map = face_result.get('gender', {})
                        # Проверка дали 'gender' е речник и не е празен
                        if isinstance(dominant_gender_map, dict) and dominant_gender_map:
                            dominant_gender = max(
                                dominant_gender_map, key=dominant_gender_map.get)
                            gender_confidence = dominant_gender_map[dominant_gender]
                        else:  # Ако DeepFace върне директно dominant_gender или празен речник
                            dominant_gender = face_result.get(
                                'dominant_gender', 'N/A')
                            gender_confidence = 0

                        dominant_emotion_map = face_result.get('emotion', {})
                        if isinstance(dominant_emotion_map, dict) and dominant_emotion_map:
                            dominant_emotion = max(
                                dominant_emotion_map, key=dominant_emotion_map.get)
                            emotion_confidence = dominant_emotion_map[dominant_emotion]
                        else:
                            dominant_emotion = face_result.get(
                                'dominant_emotion', 'N/A')
                            emotion_confidence = 0

                        region = face_result['region']

                        st.write(f"- **Възраст:** {age}")
                        st.write(
                            f"- **Пол:** {dominant_gender} ({gender_confidence:.2f}%)" if gender_confidence > 0 else f"- **Пол:** {dominant_gender}")
                        st.write(
                            f"- **Емоция:** {dominant_emotion} ({emotion_confidence:.2f}%)" if emotion_confidence > 0 else f"- **Емоция:** {dominant_emotion}")

                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        cv2.rectangle(img_cv2_annotated_bgr,
                                      (x, y), (x+w, y+h), (0, 255, 0), 2)
                        text_gender_char = dominant_gender[0].upper(
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
                    else:  # Ако num_faces_detected е 0 след обхождане на всички резултати
                        st.info(
                            "Не бяха открити лица в изображението, които да бъдат анализирани.")
# --- END OF FILE app/face_analysis_image.py ---
