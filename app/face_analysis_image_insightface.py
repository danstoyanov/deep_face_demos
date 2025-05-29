import streamlit as st
import cv2
import numpy as np
from utils import resize_image_for_display
# INSIGHTFACE_AVAILABLE се проверява в main.py преди извикване на render_page на този модул
# MAX_DISPLAY_DIM ще бъде подаден като аргумент


def render_page(insightface_available_flag, insightface_model_app, max_display_dim_passed):
    st.header("Анализ на изображение (Пол, Възраст с InsightFace)")
    st.write("Качете изображение, за да анализирате лицата за пол и възраст, използваййки модела InsightFace 'buffalo_l'.")

    if not insightface_available_flag:
        st.error("Библиотеката 'insightface' не е инсталирана. Тази функционалност не е достъпна. Моля, инсталирайте я: `pip install insightface onnxruntime`")
        st.stop()

    if insightface_model_app is None:
        st.error(
            "InsightFace моделът ('buffalo_l') не е зареден. Функционалността не е достъпна.")
        st.stop()

    uploaded_img_file = st.file_uploader(
        "Изберете изображение...", type=["jpg", "jpeg", "png"], key="insightface_image_analysis_uploader_module"
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
                img_cv2_original_bgr, max_display_dim_passed)
            img_display_resized_rgb = cv2.cvtColor(
                img_display_resized_bgr, cv2.COLOR_BGR2RGB)
            st.image(img_display_resized_rgb,
                     caption=f"Оригинално изображение ({uploaded_img_file.name})", use_column_width='auto')

            if st.button("🚀 Анализирай изображението (InsightFace)", key="analyze_image_insightface_btn"):
                with st.spinner("Анализиране на лицата с InsightFace..."):
                    try:
                        # Подаваме копие, за да не модифицираме оригиналния numpy масив
                        faces = insightface_model_app.get(
                            img_cv2_original_bgr.copy())
                    except Exception as e:
                        st.error(
                            f"Възникна грешка при InsightFace анализ: {e}")
                        faces = []  # Празен списък при грешка

                img_cv2_annotated_bgr = img_cv2_original_bgr.copy()

                if not faces:  # Проверяваме дали списъкът с лица е празен
                    st.info(
                        f"Не са намерени лица на изображението ({uploaded_img_file.name}) или възникна грешка при анализа.")
                else:
                    num_faces_processed = 0
                    for i, face_obj in enumerate(faces):
                        # Проверка дали bbox е наличен; gender и age се извличат с getattr за безопасност
                        if not hasattr(face_obj, 'bbox') or face_obj.bbox is None:
                            st.caption(
                                f"Пропускам лице {i+1} поради липсващи данни за bbox.")
                            continue

                        num_faces_processed += 1
                        st.write(f"---")
                        st.write(f"#### Лице {num_faces_processed}:")

                        # 0 for female, 1 for male. -1 if not estimated.
                        gender_val = getattr(face_obj, 'gender', -1)
                        # age value. -1 if not estimated.
                        age_val = getattr(face_obj, 'age', -1)

                        gender_str = "N/A"
                        if gender_val == 0:
                            gender_str = "Жена"
                        elif gender_val == 1:
                            gender_str = "Мъж"

                        # Показва възрастта само ако е > 0
                        age_str = str(int(age_val)) if age_val > 0 else "N/A"

                        st.write(f"- **Пол:** {gender_str}")
                        st.write(f"- **Възраст:** {age_str}")
                        # Може да се добави и det_score, ако е интересно:
                        # if hasattr(face_obj, 'det_score'):
                        #     st.write(f"- Достоверност на детекция: {face_obj.det_score:.2f}")

                        bbox = face_obj.bbox.astype(int)
                        # bbox е [x1, y1, x2, y2]
                        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

                        cv2.rectangle(img_cv2_annotated_bgr,
                                      (x1, y1), (x2, y2), (0, 255, 0), 2)

                        text_gender_char = gender_str[0].upper(
                        ) if gender_str != "N/A" else "N"
                        text_age_display = age_str if age_str != "N/A" else "?"
                        text_info = f"В: {text_age_display} П: {text_gender_char}"

                        y_text_offset = y1 - 10
                        if y_text_offset < 10:
                            y_text_offset = y1 + 15

                        cv2.putText(img_cv2_annotated_bgr, text_info, (x1, y_text_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if num_faces_processed > 0:
                        st.subheader("Анотирано изображение (InsightFace):")
                        annotated_display_resized_bgr = resize_image_for_display(
                            img_cv2_annotated_bgr, max_display_dim_passed)
                        annotated_display_resized_rgb = cv2.cvtColor(
                            annotated_display_resized_bgr, cv2.COLOR_BGR2RGB)
                        st.image(annotated_display_resized_rgb,
                                 caption=f"Анализирани лица: {num_faces_processed}", use_column_width='auto')
                    elif faces and num_faces_processed == 0:
                        st.info(
                            f"Намерени са обекти ({len(faces)}), но нито един не можа да бъде обработен като валидно лице от InsightFace на изображението ({uploaded_img_file.name}).")
                    # Случаят 'if not faces:' вече е покрит по-горе
