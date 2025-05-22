# Multifunctional Face Analysis Application (Streamlit)

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit App">
  <img src="https://img.shields.io/badge/DeepFace-Analysis-007ACC?style=for-the-badge&logo=python&logoColor=white" alt="DeepFace">
  <img src="https://img.shields.io/badge/InsightFace-Comparison-4CAF50?style=for-the-badge&logo=python&logoColor=white" alt="InsightFace">
  <img src="https://img.shields.io/badge/OpenCV-ImageProcessing-555555?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
</p>

This Streamlit web application, **developed as a university project**, provides a comprehensive set of tools for face analysis. It leverages the powerful **DeepFace** library for age, gender, and emotion detection, and **InsightFace** for robust face comparison. The application supports analysis from images, videos, and real-time webcam feeds, and features a modularized structure for better organization and maintainability.

## 🌟 Features

The application offers four main modes of operation, selectable from the sidebar:

1.  **Image Analysis (DeepFace):**
    *   Upload an image file (JPG, JPEG, PNG).
    *   Detects faces and predicts age, gender, and dominant emotion for each.
    *   Annotates the original image with bounding boxes and analysis results.
    *   Logic handled by `app/face_analysis_image.py`.

2.  **Face Comparison (InsightFace):**
    *   Upload two images.
    *   Uses InsightFace to extract high-dimensional face embeddings.
    *   Calculates the cosine similarity between the two embeddings to determine facial resemblance.
    *   Provides a similarity score and an interpretation based on a predefined threshold.
    *   Logic handled by `app/face_comparison.py`.

3.  **Video Emotion Analysis (DeepFace):**
    *   Upload a video file (MP4, AVI, MOV, MKV).
    *   Processes the video frame by frame (with an adjustable frame skip for performance).
    *   Detects faces and identifies the dominant emotion in each analyzed frame.
    *   Generates a new video file with faces annotated with their detected emotions.
    *   Presents a bar chart summary of all detected emotions throughout the video.
    *   Offers a download option for the processed video and attempts to display it in the browser.
    *   Logic handled by `app/video_emotion_analysis.py`.

4.  **Webcam Real-time Analysis (DeepFace):**
    *   Activates your local webcam for live face detection and analysis.
    *   Provides real-time estimates of age, gender, and dominant emotion for detected faces directly on the live feed.
    *   Allows selection of different DeepFace detector backends to optimize performance.
    *   Logic handled by `app/webcam_realtime_analysis.py`.

Shared utilities and configuration are managed by `app/utils.py` and `app/config.py` respectively. The main application entry point is `app/main.py`.

## 📁 Project Structure

```
your-repo-name/
├── app/
│   ├── main.py                   # Main application entry point
│   ├── config.py                 # Global configurations
│   ├── utils.py                  # Shared utility functions (model loading, image resize)
│   ├── face_analysis_image.py    # Mode 1: Image Analysis
│   ├── face_comparison.py        # Mode 2: Face Comparison
│   ├── video_emotion_analysis.py # Mode 3: Video Emotion Analysis
│   ├── webcam_realtime_analysis.py # Mode 4: Webcam Real-time Analysis
│   └── bdu_black_logo.jpg        # Example logo file
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 📸 Screenshots / Demo

*(It is highly recommended to add screenshots or a short GIF/video demonstrating the application here once it's deployed or running.)*

## 🚀 Getting Started

Follow these steps to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8 or newer
*   `pip` (Python package installer)
*   (Optional but recommended) Git for cloning the repository.

### Installation

1.  **Clone the repository (or download and extract the `app` folder and `requirements.txt`):**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
    *(Remember to replace `your-username/your-repo-name` with the actual path to your repository.)*

2.  **Navigate to the project directory:**
    Ensure you are in the directory that *contains* the `app` folder (e.g., `your-repo-name`).

3.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

4.  **Create/Verify `requirements.txt`:**
    Ensure a file named `requirements.txt` exists in the root project directory (the one containing the `app` folder). It should have the following content:

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

5.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    **Note on InsightFace (GPU vs. CPU):**
    If you have an NVIDIA GPU and CUDA Toolkit installed, uncomment `onnxruntime-gpu` and comment `onnxruntime` in `requirements.txt` for significantly faster face comparison. Otherwise, `onnxruntime` (CPU version) will be sufficient.

### Usage

1.  **Run the Streamlit application:**
    From the root project directory (the one containing the `app` folder), run:

    ```bash
    streamlit run app/main.py
    ```

2.  **Access the application:**
    Your default web browser will automatically open to the Streamlit application, usually at `http://localhost:8501`.

3.  **Initial Model Downloads:**
    The first time you run the application, DeepFace and InsightFace models will be downloaded and cached. This process might take some time, especially for DeepFace models. Progress messages will be displayed in the Streamlit interface.

## ⚙️ Configuration and Notes

*   **Model Caching:** DeepFace and InsightFace models are loaded and cached using `st.cache_resource` in `app/utils.py`. This ensures models are loaded only once, improving performance.
*   **InsightFace Providers:** The InsightFace model attempts to use `CUDAExecutionProvider` (GPU) first, falling back to `CPUExecutionProvider` if GPU is unavailable or fails.
*   **Video Processing:**
    *   Temporary files are used for input and output videos.
    *   The `app/config.py` file allows switching between `MP4 (mp4v)` and `AVI (XVID)` for the output video format, which can be useful if one format has playback issues in the browser.
*   **Webcam Performance:** Real-time webcam analysis can be computationally intensive. Performance depends on hardware and the chosen `detector_backend`.
*   **`detector_backend`:** Various DeepFace detectors can be selected in the sidebar for video and webcam modes:
    *   `'opencv'`: Generally fastest.
    *   `'ssd'`, `'mtcnn'`, `'retinaface'`, `'yunet'`: Offer different balances of speed and accuracy.
*   **Logo Display:** The application attempts to load a logo from `app/bdu_black_logo.jpg`. Path and display width are configurable in `app/config.py`.

## 💡 Troubleshooting

*   **`ModuleNotFoundError`:** Ensure all dependencies from `requirements.txt` are installed within your active virtual environment.
*   **`SyntaxError: unterminated string literal`**: This can occur if f-strings in the Python code are accidentally broken across multiple lines during copy-pasting. Ensure long f-strings are on a single line.
*   **Camera Not Opening:**
    *   Check webcam connection and ensure it's enabled.
    *   Verify no other apps are using the webcam.
    *   Grant browser/system camera permissions.
*   **Model Download Errors:**
    *   Check internet connection.
    *   Try restarting the application.
*   **Video Processing/Playback Issues:**
    *   Ensure your `opencv-python` installation has `ffmpeg` support (often included, but might need separate `ffmpeg` installation on some systems).
    *   Try switching the video output format in `app/config.py` (e.g., from MP4 to AVI or vice-versa).
    *   The browser might not support the specific video codec/container combination generated.
*   **InsightFace Model Loading Fails (GPU):**
    *   Ensure correct NVIDIA drivers, CUDA Toolkit, and cuDNN versions compatible with `onnxruntime-gpu`. Otherwise, use the CPU version (`onnxruntime`).

## 🤝 Contributing

Contributions, suggestions, and issue reports are welcome! Please open an issue or submit a pull request.

## 📄 License

This project is open-source and available under the [MIT License](LICENSE). *(You should create a `LICENSE` file in your repository if you haven't already, e.g., by copying one from choosealicense.com).*

## 🙏 Acknowledgments

This application originated as a **university project** by the Department of Computer Systems and Technologies team at Burgas State University (BSU) / Бургаски държавен университет (БДУ).
