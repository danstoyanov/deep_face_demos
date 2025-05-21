# --- START OF FILE app/config.py ---
MAX_DISPLAY_DIM = 600
# Уверете се, че този файл съществува в същата директория като main.py или посочете правилния път
LOGO_PATH = "bdu_black_logo.jpg"
LOGO_DISPLAY_WIDTH = 100

# Constants for face comparison
INSIGHTFACE_THRESHOLD = 0.58

# Constants for video analysis
DEFAULT_VIDEO_FPS = 25.0

# --- Настройки за видео кодек ---
# Пробвайте да промените USE_AVI_XVID на True, ако имате проблеми с MP4 плейването в браузъра
USE_AVI_XVID = True  # Нагласете на True, за да използвате AVI/XVID

if USE_AVI_XVID:
    VIDEO_OUTPUT_CODEC = 'XVID'
    VIDEO_OUTPUT_SUFFIX = "_emotions_processed.avi"
    VIDEO_MIME_TYPE = "video/x-msvideo"
    TEMP_FILE_SUFFIX = "_processed.avi"
else:
    VIDEO_OUTPUT_CODEC = 'mp4v'  # или 'avc1' ако е налично и по-добро за MP4
    VIDEO_OUTPUT_SUFFIX = "_emotions_processed.mp4"
    VIDEO_MIME_TYPE = "video/mp4"
    TEMP_FILE_SUFFIX = "_processed.mp4"
# --- END OF FILE app/config.py ---
