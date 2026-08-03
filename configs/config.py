from pathlib import Path

# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "datasets" / "CMAPSSData"

TRAIN_FILE = DATASET_DIR / "train_FD001.txt"
TEST_FILE = DATASET_DIR / "test_FD001.txt"
RUL_FILE = DATASET_DIR / "RUL_FD001.txt"

OUTPUT_DIR = BASE_DIR / "output"

FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"

# ==========================================================
# Visualization
# ==========================================================

FIG_DPI = 300

# ==========================================================
# Noise Filter
# ==========================================================

SAVGOL_WINDOW = 11
SAVGOL_POLYORDER = 2

# ==========================================================
# Z-Score
# ==========================================================

ZSCORE_THRESHOLD = 3.0

# ==========================================================
# Kalman Filter Parameters
# ==========================================================

KF_PROCESS_VARIANCE = 1e-5
KF_MEASUREMENT_VARIANCE = 1e-2
KF_INITIAL_ERROR = 1.0

# ==========================================================
# Sliding Window
# ==========================================================

WINDOW_SIZE = 30

# ==========================================================
# Random Seed
# ==========================================================

RANDOM_STATE = 42