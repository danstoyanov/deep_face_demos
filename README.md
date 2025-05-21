Разбира се, ето една подобрена версия на `README.md` файла за вашия GitHub репозиториум. Включил съм стандартни секции за GitHub, както и няколко допълнителни детайла и съвети:

---

# Multifunctional Face Analysis Application (Streamlit)

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit App">
  <img src="https://img.shields.io/badge/DeepFace-Analysis-007ACC?style=for-the-badge&logo=python&logoColor=white" alt="DeepFace">
  <img src="https://img.shields.io/badge/InsightFace-Comparison-4CAF50?style=for-the-badge&logo=python&logoColor=white" alt="InsightFace">
  <img src="https://img.shields.io/badge/OpenCV-ImageProcessing-555555?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
</p>

This Streamlit web application provides a comprehensive set of tools for face analysis, leveraging the powerful **DeepFace** library for age, gender, and emotion detection, and **InsightFace** for robust face comparison. It supports analysis from images, videos, and real-time webcam feeds.

## 🌟 Features

The application offers four main modes of operation, selectable from the sidebar:

1.  **Image Analysis (DeepFace):**
    *   Upload an image file (JPG, JPEG, PNG).
    *   Detects faces and predicts age, gender, and dominant emotion for each.
    *   Annotates the original image with bounding boxes and analysis results.

2.  **Face Comparison (InsightFace):**
    *   Upload two images.
    *   Uses InsightFace to extract high-dimensional face embeddings.
    *   Calculates the cosine similarity between the two embeddings to determine facial resemblance.
    *   Provides a similarity score and an interpretation based on a predefined threshold.

3.  **Video Emotion Analysis (DeepFace):**
    *   Upload a video file (MP4, AVI, MOV, MKV).
    *   Processes the video frame by frame (with an adjustable frame skip for performance).
    *   Detects faces and identifies the dominant emotion in each analyzed frame.
    *   Generates a new video file with faces annotated with their detected emotions.
    *   Presents a bar chart summary of all detected emotions throughout the video.
    *   Offers a download option for the processed video.

4.  **Webcam Real-time Analysis (DeepFace):**
    *   Activates your local webcam for live face detection and analysis.
    *   Provides real-time estimates of age, gender, and dominant emotion for detected faces directly on the live feed.
    *   Allows selection of different DeepFace detector backends to optimize performance based on your hardware.

## 📸 Screenshots / Demo

*(It is highly recommended to add screenshots or a short GIF/video demonstrating the application here once it's deployed or running.)*

## 🚀 Getting Started

Follow these steps to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8 or newer
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
    *(Remember to replace `your-username/your-repo-name` with the actual path to your repository.)*

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Create `requirements.txt`:**
    Create a file named `requirements.txt` in the root directory of your project (where `second_streamlit_deepface_app.py` is located) and paste the following content into it:

    ```
    streamlit>=1.30.0
    deepface
    opencv-python>=4.5.0
    numpy>=1.20.0
    Pillow>=9.0.0
    
    # Required for the Face Comparison (InsightFace) feature:
    # Choose either onnxruntime (CPU) or onnxruntime-gpu (GPU)
    insightface
    onnxruntime # For CPU-only execution (default)
    # onnxruntime-gpu # Uncomment this and comment the line above for GPU acceleration (requires NVIDIA CUDA toolkit)
    scikit-learn
    ```

4.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    **Note on InsightFace (GPU vs. CPU):**
    If you have an NVIDIA GPU and CUDA Toolkit installed, uncomment `onnxruntime-gpu` and comment `onnxruntime` in `requirements.txt` for significantly faster face comparison. Otherwise, `onnxruntime` (CPU version) will be sufficient.

### Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run second_streamlit_deepface_app.py
    ```

2.  **Access the application:**
    Your default web browser will automatically open to the Streamlit application, usually at `http://localhost:8501`.

3.  **Initial Model Downloads:**
    The first time you run the application, DeepFace and InsightFace models will be downloaded and cached. This process might take some time, especially for DeepFace models. A progress message will be displayed in the Streamlit interface.

## ⚙️ Configuration and Notes

*   **Model Caching:** DeepFace and InsightFace models are loaded and cached using `st.cache_resource`. This ensures that models are loaded only once upon application startup or their first use, significantly improving subsequent performance within the same session.
*   **InsightFace Providers:** The InsightFace model attempts to use `CUDAExecutionProvider` first if available for GPU acceleration. If not successful, it gracefully falls back to `CPUExecutionProvider`.
*   **Video Processing:** For video analysis, temporary files are created for both the input video and the processed output video. The input video temporary file is cleaned up after processing. The output video temporary file remains until the application is closed or a new video is uploaded, allowing for download.
*   **Webcam Performance:** Real-time webcam analysis can be computationally intensive. Performance depends heavily on your hardware and the chosen `detector_backend` (e.g., `opencv` is often faster than `retinaface` for real-time).
*   **`detector_backend`:** DeepFace allows different face detectors.
    *   `'opencv'`: Fastest, but might be less accurate for smaller or angled faces.
    *   `'ssd'`: A good balance of speed and accuracy.
    *   `'mtcnn'`, `'retinaface'`, `'yunet'`: Generally more accurate but slower, especially for real-time applications. `yunet` often offers a good balance of accuracy and speed.
*   **Logo Display:** The application attempts to load a logo from a file named `bdu_black_logo.jpg` located in the same directory as the script. If you wish to display your own logo, replace this file with your image (or update the `LOGO_PATH` variable in the script).

## 💡 Troubleshooting

*   **`ModuleNotFoundError`:** Ensure all dependencies from `requirements.txt` are installed. Activate your virtual environment before running `pip install -r requirements.txt`.
*   **Camera Not Opening:**
    *   Check if your webcam is connected and enabled.
    *   Ensure no other applications are using the webcam.
    *   Grant camera permissions to your browser or system if prompted.
*   **Model Download Errors:**
    *   Verify your internet connection.
    *   Sometimes, temporary network issues can cause failures. Try restarting the application.
*   **Video Processing Issues:**
    *   Ensure your `opencv-python` installation has `ffmpeg` support. On some systems, this might require manual installation of `ffmpeg`.
    *   Check if the video file is corrupted or in an unsupported format.
*   **InsightFace Model Loading Fails:**
    *   If you're trying to use `onnxruntime-gpu`, ensure your NVIDIA drivers, CUDA Toolkit, and cuDNN are correctly installed and compatible with your `onnxruntime-gpu` version. Otherwise, stick to `onnxruntime` (CPU version).

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements or find any issues, please open an issue or submit a pull request.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE). *(You should create a `LICENSE` file in your repository if you haven't already.)*

## 🙏 Acknowledgments

This application was developed by the Department of Computer Systems and Technologies team at Burgas Free University (BFU) / Бургаски свободен университет (БСУ).

---
