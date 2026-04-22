import os

PACKAGE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

DATASET_DIR = os.path.join(PACKAGE_DIR, "data", "src")

SERVER_HOST = "127.0.0.1:5001"

LOW_THRESHOLD = 0.1
HIGH_THRESHOLD = 0.6

OBSERVATION_WINDOW = 3

ROI_PAD_RATIO = 0.1
ROI_PAD_PIX = 8

CACHE_DIR = os.path.join(PACKAGE_DIR, "edge", "cache")
TEMP_DIR = os.path.join(PACKAGE_DIR, "edge", "temp")
RESULT_BW_CSV = os.path.join(PACKAGE_DIR, "edge", "result_bw.csv")
RESULT_BW_TXT = os.path.join(PACKAGE_DIR, "edge", "result_bw.txt")

FEATURES_CSV = os.path.join(PACKAGE_DIR, "edge", "features.csv")

USE_ANFIS = True
ANFIS_WEIGHTS = os.path.join(PACKAGE_DIR, "models", "anfis_weights.json")

BACKGROUND_ENCODING_QP = 28

USE_COMPOSITE = True
COMPOSITE_BG_QUALITY = 10
COMPOSITE_QUALITY = 30

USE_YOLO = False

VIDEO_BATCH_SIZE = 5

ENCODING_QP = 5

FFMPEG_BIN = os.environ.get("FFMPEG_BIN", "ffmpeg")
