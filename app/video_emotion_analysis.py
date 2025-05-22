import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
import os
import tempfile
import time
import traceback
# Импорт на константи от config.py, които вече ще включват логиката за избор на кодек
from config import DEFAULT_VIDEO_FPS, VIDEO_OUTPUT_CODEC, VIDEO_OUTPUT_SUFFIX, VIDEO_MIME_TYPE, TEMP_FILE_SUFFIX


def render_page(deepface_models_loaded):
    st.header("3. Анализ на емоции във видео (DeepFace)")
    if not deepface_models_loaded:
        st.error("DeepFace моделите не са заредени. Функционалността не е достъпна.")
        st.stop()

    uploaded_video_file = st.file_uploader(
        "Изберете видео файл (.mp4, .avi, .mov, .mkv)...", type=["mp4", "avi", "mov", "mkv"], key="deepface_video_uploader_module"
    )

    frame_skip_file = st.sidebar.slider(
        "Пропускай кадри за анализ (файл):", 0, 10, 1, key="video_frame_skip_file_module_active")
    detector_backend_video_file = st.sidebar.selectbox(
        "DeepFace детектор за видео (файл):",
        ('opencv', 'ssd', 'mtcnn', 'retinaface', 'yunet'), index=0, key="video_detector_file_module_active")

    if uploaded_video_file is not None:
        current_video_id = f"{uploaded_video_file.name}_{uploaded_video_file.size}"
        input_video_session_key = f"temp_video_path_{current_video_id}"
        # Добавяме кодека към ключа
        output_video_display_cache_key = f"cached_output_video_path_{current_video_id}_{detector_backend_video_file}_{frame_skip_file}_{VIDEO_OUTPUT_CODEC}"

        if input_video_session_key not in st.session_state or \
           not os.path.exists(st.session_state[input_video_session_key]):
            try:
                # Уверяваме се, че старият файл се изтрива, ако ключът съществува, но файлът не
                if input_video_session_key in st.session_state and st.session_state.get(input_video_session_key):
                    if os.path.exists(st.session_state[input_video_session_key]):
                        try:
                            os.remove(
                                st.session_state[input_video_session_key])
                        except OSError:
                            pass  # Игнорираме грешка при изтриване
                    del st.session_state[input_video_session_key]

                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_video_file.name.split('.')[-1]}") as tfile:
                    tfile.write(uploaded_video_file.read())
                    st.session_state[input_video_session_key] = tfile.name
                # st.caption(f"Входното видео е запазено временно в: {st.session_state[input_video_session_key]}") # Може да се махне за по-чист интерфейс
            except Exception as e:
                st.error(f"Грешка при запис на качения видео файл: {e}")
                st.stop()

        temp_video_path = st.session_state[input_video_session_key]

        st.subheader("Оригинално видео:")
        col_orig_video, _ = st.columns([2, 1])
        with col_orig_video:
            if os.path.exists(temp_video_path):
                st.video(temp_video_path)
            else:
                st.warning(
                    "Оригиналният видео файл не е намерен. Моля, качете го отново.")
        # st.caption("Оригиналното видео се показва с автоматично оразмеряване.")

        if st.button("🚀 Анализирай видеото (DeepFace)", key="analyze_video_deepface_btn"):
            if not os.path.exists(temp_video_path):
                st.error(
                    "Грешка: Входният видео файл не е намерен. Моля, качете го отново.")
                st.stop()

            base_name, _ = os.path.splitext(uploaded_video_file.name)
            # Името на файла за сваляне вече използва VIDEO_OUTPUT_SUFFIX от config.py
            output_video_filename_for_download = f"{base_name}{VIDEO_OUTPUT_SUFFIX}"

            current_run_output_video_path = None
            try:
                # Суфиксът на временния файл вече идва от config.py (TEMP_FILE_SUFFIX)
                with tempfile.NamedTemporaryFile(delete=False, suffix=TEMP_FILE_SUFFIX) as t_out:
                    current_run_output_video_path = t_out.name
                st.info(
                    f"Обработката ще запише резултата в нов временен файл: {current_run_output_video_path} (Кодек: {VIDEO_OUTPUT_CODEC})")
            except Exception as e_tempfile:
                st.error(
                    f"Грешка при създаване на временен изходен файл: {e_tempfile}")
                st.stop()

            if not current_run_output_video_path:  # Двойна проверка
                st.error("Не можа да се създаде временен изходен файл.")
                st.stop()

            cap = None
            out_writer = None
            analysis_successful = False

            try:
                cap = cv2.VideoCapture(temp_video_path)
                if not cap.isOpened():
                    st.error(
                        f"Грешка: Не може да се отвори видео файлът: {uploaded_video_file.name}")
                    st.stop()

                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not (fps and 0 < fps < 200):
                    st.warning(
                        f"Невалидна FPS стойност ({fps}) от видеото. Задавам {DEFAULT_VIDEO_FPS} FPS.")
                    fps = DEFAULT_VIDEO_FPS

                fourcc = cv2.VideoWriter_fourcc(*VIDEO_OUTPUT_CODEC)
                out_writer = cv2.VideoWriter(
                    current_run_output_video_path, fourcc, fps, (frame_width, frame_height))

                if not out_writer.isOpened():
                    st.error(
                        f"Грешка: Не може да се инициализира VideoWriter с кодек '{VIDEO_OUTPUT_CODEC}'. Проверете вашата OpenCV/FFmpeg инсталация и наличните кодеци.")
                    if os.path.exists(current_run_output_video_path):
                        # Изтриваме неуспешния файл
                        os.remove(current_run_output_video_path)
                    st.stop()

                # st.info(f"Обработка на видео: {uploaded_video_file.name} ({frame_width}x{frame_height} @ {fps:.2f} FPS) с детектор '{detector_backend_video_file}'. Кодек: {VIDEO_OUTPUT_CODEC}")
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
                last_known_faces_data = []

                while cap.isOpened():
                    ret, frame_bgr = cap.read()
                    if not ret:
                        break

                    current_frame_num_read += 1
                    frame_to_write = frame_bgr.copy()

                    if current_frame_num_read % (frame_skip_file + 1) == 1:
                        processed_frame_count_for_analysis += 1
                        try:
                            results = DeepFace.analyze(frame_bgr.copy(), actions=['emotion'], enforce_detection=True,
                                                       detector_backend=detector_backend_video_file, silent=True)
                            current_faces_data = []
                            if isinstance(results, list) and len(results) > 0:
                                for result in results:
                                    if 'region' in result and result['region']['w'] > 0 and result['region']['h'] > 0:
                                        x, y, w, h = result['region']['x'], result['region'][
                                            'y'], result['region']['w'], result['region']['h']
                                        dominant_emotion_map = result.get(
                                            'emotion', {})
                                        if isinstance(dominant_emotion_map, dict) and dominant_emotion_map:
                                            emotion = max(
                                                dominant_emotion_map, key=dominant_emotion_map.get)
                                            confidence = dominant_emotion_map[emotion]
                                        else:
                                            emotion = result.get(
                                                'dominant_emotion', "N/A")
                                            confidence = 0.0
                                        emotions_summary[emotion] = emotions_summary.get(
                                            emotion, 0) + 1
                                        current_faces_data.append({'box': (
                                            x, y, w, h), 'text': f"{emotion} ({confidence:.2f}%)" if confidence > 0 else emotion})
                            last_known_faces_data = current_faces_data
                        except ValueError:
                            last_known_faces_data = []
                        except Exception as deepface_e:
                            st.warning(
                                f"DeepFace грешка при анализ на кадър {current_frame_num_read}: {deepface_e}")
                            last_known_faces_data = []

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

                end_time = time.time()
                progress_bar.empty()
                status_text.empty()
                percent_complete_text.empty()
                st.success(
                    f"Видео обработката приключи за {end_time - start_time:.2f} секунди.")
                st.info(
                    f"Общо прочетени кадри: {current_frame_num_read}. Анализирани: {processed_frame_count_for_analysis}")
                analysis_successful = True

            except Exception as e_video:
                st.error(
                    f"Възникна неочаквана грешка по време на видео обработката: {e_video}")
                st.error(traceback.format_exc())
                analysis_successful = False
            finally:
                if cap is not None:
                    cap.release()
                if out_writer is not None:
                    out_writer.release()

            if analysis_successful and current_run_output_video_path and \
               os.path.exists(current_run_output_video_path) and \
               os.path.getsize(current_run_output_video_path) > 0:

                st.subheader("Обработено видео (току-що генерирано):")
                st.caption(
                    f"Файл за показване: {current_run_output_video_path}, Размер: {os.path.getsize(current_run_output_video_path) / (1024*1024):.2f} MB")

                # Изрична проверка преди st.video
                if os.path.exists(current_run_output_video_path) and os.path.getsize(current_run_output_video_path) > 0:
                    col_proc_video, _ = st.columns([2, 1])
                    with col_proc_video:
                        st.video(current_run_output_video_path,
                                 format=VIDEO_MIME_TYPE)  # Подаваме MIME типа

                    # Успешно е, запазваме пътя в сесията за кеширане на показването
                    if output_video_display_cache_key in st.session_state and \
                       st.session_state.get(output_video_display_cache_key) and \
                       os.path.exists(st.session_state[output_video_display_cache_key]) and \
                       st.session_state[output_video_display_cache_key] != current_run_output_video_path:
                        try:
                            os.remove(
                                st.session_state[output_video_display_cache_key])
                        except OSError:
                            pass
                    st.session_state[output_video_display_cache_key] = current_run_output_video_path

                    with open(current_run_output_video_path, 'rb') as video_file_for_download:
                        video_bytes_for_download = video_file_for_download.read()
                    st.download_button(label=f"📥 Свали обработеното видео ({VIDEO_OUTPUT_CODEC.upper()})",
                                       data=video_bytes_for_download,
                                       file_name=output_video_filename_for_download,
                                       mime=VIDEO_MIME_TYPE,
                                       key=f"download_btn_current_{output_video_display_cache_key}")
                else:
                    st.error(
                        "Грешка: Генерираният файл изглежда невалиден или празен точно преди показване.")

            else:  # analysis_successful is False OR file is invalid
                st.error(
                    "Неуспешно генериране на изходното видео или файлът е празен/невалиден.")
                if current_run_output_video_path and os.path.exists(current_run_output_video_path):
                    try:
                        os.remove(current_run_output_video_path)
                        st.info(
                            f"Изтрит неуспешно генериран файл: {current_run_output_video_path}")
                    except OSError:
                        pass

            if emotions_summary and analysis_successful:
                st.subheader("Обобщение на засечените доминиращи емоции:")
                sorted_emotions = dict(
                    sorted(emotions_summary.items(), key=lambda item: item[1], reverse=True))
                st.bar_chart(sorted_emotions)
            elif not analysis_successful:
                st.info(
                    "Обобщение на емоциите не е налично поради грешка в обработката.")
            else:
                st.info("Не са засечени емоции по време на обработката.")

        elif output_video_display_cache_key in st.session_state and \
                st.session_state.get(output_video_display_cache_key) and \
                os.path.exists(st.session_state[output_video_display_cache_key]) and \
                os.path.getsize(st.session_state[output_video_display_cache_key]) > 0:

            cached_video_path = st.session_state[output_video_display_cache_key]
            st.subheader("Предишно обработено видео (кеширано):")
            st.caption(
                f"Кеширан файл: {cached_video_path}, Кодек: {VIDEO_OUTPUT_CODEC.upper()}")

            base_name, _ = os.path.splitext(uploaded_video_file.name)
            output_video_filename_for_download = f"{base_name}{VIDEO_OUTPUT_SUFFIX}"

            col_cached_proc_video, _ = st.columns([2, 1])
            with col_cached_proc_video:
                st.video(cached_video_path, format=VIDEO_MIME_TYPE)

            try:
                with open(cached_video_path, 'rb') as video_file_for_download:
                    video_bytes_for_download = video_file_for_download.read()
                st.download_button(label=f"📥 Свали това (кеширано) видео ({VIDEO_OUTPUT_CODEC.upper()})",
                                   data=video_bytes_for_download,
                                   file_name=output_video_filename_for_download,
                                   mime=VIDEO_MIME_TYPE,
                                   key=f"download_btn_cached_{output_video_display_cache_key}")
            except FileNotFoundError:
                st.error("Кешираният видео файл не беше намерен за сваляне.")
            # Обобщението на емоциите не се показва за кешираното видео, тъй като не се пази в сесията.
