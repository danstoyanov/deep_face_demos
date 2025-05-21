# --- START OF FILE app/face_comparison.py ---
import streamlit as st
import cv2
import numpy as np
from utils import resize_image_for_display, INSIGHTFACE_AVAILABLE  # Импорти от utils.py
# MAX_DISPLAY_DIM и INSIGHTFACE_THRESHOLD ще се подадат като аргументи

# sklearn.metrics.pairwise.cosine_similarity се импортира условно по-долу


def render_page(INSIGHTFACE_AVAILABLE, insightface_model_app, MAX_DISPLAY_DIM, INSIGHTFACE_THRESHOLD):
    st.header("2. Сравняване на две лица (Cosine Similarity с InsightFace)")

    if not INSIGHTFACE_AVAILABLE:
        st.error("Библиотеките 'insightface' или 'scikit-learn' не са инсталирани. Тази функционалност не е достъпна. Моля, инсталирайте ги: `pip install insightface onnxruntime scikit-learn`")
        st.stop()

    if insightface_model_app is None:
        st.error("InsightFace моделът не е зареден. Функционалността не е достъпна.")
        st.stop()

    # Условно импортиране на cosine_similarity
    if INSIGHTFACE_AVAILABLE:
        from sklearn.metrics.pairwise import cosine_similarity

    col1, col2 = st.columns(2)
    with col1:
        uploaded_img1_file = st.file_uploader(
            "Изображение 1:", type=["jpg", "jpeg", "png"], key="insightface_img1_module")
    with col2:
        uploaded_img2_file = st.file_uploader(
            "Изображение 2:", type=["jpg", "jpeg", "png"], key="insightface_img2_module")

    def get_face_embedding_and_draw_insight(image_file, img_identifier_str, insight_app_model_instance):
        # Тази функция се извиква само ако image_file не е None
        file_bytes_img = np.asarray(
            bytearray(image_file.read()), dtype=np.uint8)
        img_cv_original_bgr = cv2.imdecode(file_bytes_img, cv2.IMREAD_COLOR)

        if img_cv_original_bgr is None:
            st.warning(
                f"Неуспешно зареждане на Изображение {img_identifier_str}.")
            return None, None, image_file.name

        faces = insight_app_model_instance.get(img_cv_original_bgr)
        img_cv_annotated_bgr = img_cv_original_bgr.copy()

        if not faces:
            st.info(
                f"Не са намерени лица на Изображение {img_identifier_str} ({image_file.name}).")
            return None, img_cv_annotated_bgr, image_file.name

        main_face = faces[0]  # Използваме първото намерено лице
        if len(faces) > 1:
            st.caption(
                f"Намерени {len(faces)} лица в Изображение {img_identifier_str}, използва се първото.")

        bbox = main_face.bbox.astype(int)
        cv2.rectangle(img_cv_annotated_bgr,
                      (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        embedding = main_face.embedding
        return embedding, img_cv_annotated_bgr, image_file.name

    if uploaded_img1_file and uploaded_img2_file:
        if st.button("🚀 Сравни лицата (InsightFace)", key="compare_faces_insightface_btn"):
            with st.spinner("Обработка на изображенията и изчисляване на сходство..."):
                embedding1, img1_processed_bgr, filename1 = get_face_embedding_and_draw_insight(
                    uploaded_img1_file, "1", insightface_model_app)
                embedding2, img2_processed_bgr, filename2 = get_face_embedding_and_draw_insight(
                    uploaded_img2_file, "2", insightface_model_app)

                col_disp1, col_disp2 = st.columns(2)
                display_dim_comparison = MAX_DISPLAY_DIM // 2 + 100
                if display_dim_comparison > MAX_DISPLAY_DIM:
                    display_dim_comparison = MAX_DISPLAY_DIM

                if img1_processed_bgr is not None:
                    with col_disp1:
                        resized_img1_bgr = resize_image_for_display(
                            img1_processed_bgr, display_dim_comparison)
                        resized_img1_rgb = cv2.cvtColor(
                            resized_img1_bgr, cv2.COLOR_BGR2RGB)
                        st.image(
                            resized_img1_rgb, caption=f"Изображение 1: {filename1}", use_column_width='auto')

                if img2_processed_bgr is not None:
                    with col_disp2:
                        resized_img2_bgr = resize_image_for_display(
                            img2_processed_bgr, display_dim_comparison)
                        resized_img2_rgb = cv2.cvtColor(
                            resized_img2_bgr, cv2.COLOR_BGR2RGB)
                        st.image(
                            resized_img2_rgb, caption=f"Изображение 2: {filename2}", use_column_width='auto')

                if embedding1 is not None and embedding2 is not None:
                    similarity_score = cosine_similarity(
                        embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]
                    st.subheader(f"Резултат от сравнението:")
                    st.metric(label="Cosine Similarity",
                              value=f"{similarity_score:.4f}")

                    if similarity_score >= INSIGHTFACE_THRESHOLD:
                        st.success(
                            f"Лицата са ВЕРОЯТНО на един и същи човек (сходство >= {INSIGHTFACE_THRESHOLD}).")
                    else:
                        st.warning(
                            f"Лицата са ВЕРОЯТНО на различни хора (сходство < {INSIGHTFACE_THRESHOLD}).")
                    st.caption(
                        f"(Използван праг: {INSIGHTFACE_THRESHOLD} за модел 'buffalo_l'. Стойностите са между -1 и 1. По-висока = по-голямо сходство.)")
                elif (uploaded_img1_file and uploaded_img2_file) and (embedding1 is None or embedding2 is None):
                    st.error(
                        "Не можа да се извлече лице от едното или двете изображения. Сравнението е невъзможно.")
# --- END OF FILE app/face_comparison.py ---
