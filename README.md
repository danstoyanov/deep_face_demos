# Multifunctional Face Analysis & LLM Chat Application (Streamlit)

<p align="center">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit App">
  <img src="https://img.shields.io/badge/DeepFace-Analysis-007ACC?style=for-the-badge&logo=python&logoColor=white" alt="DeepFace">
  <img src="https://img.shields.io/badge/InsightFace-Comparison-4CAF50?style=for-the-badge&logo=python&logoColor=white" alt="InsightFace">
  <img src="https://img.shields.io/badge/Llama%20CPP-LLM%20Chat-9C27B0?style=for-the-badge&logo=python&logoColor=white" alt="Llama CPP LLM">
  <img src="https://img.shields.io/badge/OpenCV-Processing-555555?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Version">
</p>

This Streamlit web application, **developed as a university project**, provides a comprehensive set of tools for face analysis and an interactive chat experience with a local Large Language Model (LLM). It leverages:
*   **DeepFace** for age, gender, and emotion detection.
*   **InsightFace** for robust face comparison.
*   **Llama CPP (Phi-3 Mini model)** for conversational AI.

The application supports analysis from images, videos, real-time webcam feeds, and features a modularized structure for better organization and maintainability.

## 🌟 Features

The application offers five main modes of operation, selectable from the sidebar:

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

5.  **LLM Chat (Phi-3 Mini via Llama CPP):**
    *   Interactive chat interface with the `Phi-3-mini-4k-instruct-q4.gguf` model.
    *   Requires the model file to be downloaded and placed in the `app/models/` directory.
    *   Utilizes `llama-cpp-python` for running the GGUF model locally.
    *   Logic handled by `app/language_model_chat.py`.

Shared utilities and configuration are managed by `app/utils.py` and `app/config.py` respectively. The main application entry point is `app/main.py`.

## 📁 Project Structure

```
your-repo-name/
├── app/
│   ├── main.py                       # Main application entry point
│   ├── config.py                     # Global configurations
│   ├── utils.py                      # Shared utility functions
│   ├── face_analysis_image.py        # Mode 1: Image Analysis
│   ├── face_comparison.py            # Mode 2: Face Comparison
│   ├── video_emotion_analysis.py     # Mode 3: Video Emotion Analysis
│   ├── webcam_realtime_analysis.py   # Mode 4: Webcam Real-time Analysis
│   ├── language_model_chat.py        # Mode 5: LLM Chat
│   ├── models/                         # Directory for LLM models
│   │   └── Phi-3-mini-4k-instruct-q4.gguf # Example LLM model file (needs to be downloaded)
│   └── bdu_black_logo.jpg            # Example logo file
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 📸 Screenshots / Demo

*(It is highly recommended to add screenshots or a short GIF/video demonstrating the application here once it's deployed or running, showcasing all features including the LLM chat.)*

## 🚀 Getting Started

Follow these steps to get a copy of the project up and running on your local machine.

### Prerequisites

*   Python 3.8 or newer
*   `pip` (Python package installer)
*   (Optional but recommended) Git for cloning the repository.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```
    *(Replace `your-username/your-repo-name` with the actual path.)*

2.  **Download the LLM Model:**
    *   Download the `Phi-3-mini-4k-instruct-q4.gguf` model from [Hugging Face (microsoft/Phi-3-mini-4k-instruct-gguf)](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/blob/main/Phi-3-mini-4k-instruct-q4.gguf).
    *   Create a directory `app/models/` inside your project.
    *   Place the downloaded `.gguf` file into the `app/models/` directory.

3.  **Navigate to the project directory:**
    Ensure you are in the root directory that *contains* the `app` folder.

4.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

5.  **Update/Verify `requirements.txt`:**
    Ensure your `requirements.txt` file includes `llama-cpp-python`:
    ```
    streamlit>=1.30.0
    deepface
    opencv-python>=4.5.0
    numpy>=1.20.0
    Pillow>=9.0.0
    
    # For Face Comparison (InsightFace)
    insightface
    onnxruntime # For CPU
    # onnxruntime-gpu # For GPU
    scikit-learn

    # For LLM Chat
    llama-cpp-python>=0.2.0 
    ```

6.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    **Note on `llama-cpp-python` (GPU vs. CPU):**
    By default, the LLM will run on the CPU. For GPU acceleration with `llama-cpp-python` (if you have a compatible NVIDIA GPU and CUDA installed), you'll need to compile `llama-cpp-python` with GPU support. This usually involves setting environment variables like `CMAKE_ARGS="-DLLAMA_CUBLAS=on"` before `pip install`. Refer to the [llama-cpp-python documentation](https://github.com/abetlen/llama-cpp-python#installation-with-hardware-acceleration) for detailed instructions. After successful GPU-enabled installation, you can adjust `n_gpu_layers` in `app/language_model_chat.py`.

### Usage

1.  **Run the Streamlit application:**
    From the root project directory, run:
    ```bash
    streamlit run app/main.py
    ```

2.  **Access the application:**
    Open `http://localhost:8501` in your web browser.

3.  **Initial Model Downloads:**
    *   DeepFace and InsightFace models will be downloaded on first use if not already cached.
    *   The LLM (Phi-3 Mini) will be loaded from your local `app/models/` directory when you select the "Чат с Езиков Модел" mode. This initial loading might take some time.

## ⚙️ Configuration and Notes

*   **Model Caching:** DeepFace, InsightFace, and LLM models are loaded and cached using `st.cache_resource` to improve performance on subsequent uses within the same session.
*   **LLM Configuration:**
    *   The LLM model path and name are defined in `app/config.py`.
    *   CPU/GPU offloading for the LLM (`n_gpu_layers`) is configured in `app/language_model_chat.py` (defaults to CPU).
*   **Video Processing:** Output format (MP4/AVI) is configurable in `app/config.py`.
*   **Logo Display:** Configurable via `app/config.py`.

## 💡 Troubleshooting

*   **`ModuleNotFoundError`:** Ensure all dependencies from `requirements.txt` are installed in your active virtual environment.
*   **LLM Model Not Found:** Verify the `.gguf` file is correctly placed in `app/models/` and that `LLM_MODEL_NAME` and `LLM_MODEL_DIR` in `app/config.py` match.
*   **`llama-cpp-python` errors:**
    *   If using GPU, ensure proper compilation with GPU support.
    *   If using CPU, ensure `n_gpu_layers` is set to `0` in `app/language_model_chat.py`.
    *   Installation can sometimes be tricky; refer to the official `llama-cpp-python` documentation.
*   **Other issues:** Refer to the troubleshooting points in previous README versions for face analysis features.

## 🤝 Contributing

Contributions, suggestions, and issue reports are welcome!

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

This application originated as a **university project** by the Department of Computer Systems and Technologies team at Burgas State University (BSU) / Бургаски държавен университет (БДУ). It integrates powerful open-source libraries like DeepFace, InsightFace, and Llama CPP.
