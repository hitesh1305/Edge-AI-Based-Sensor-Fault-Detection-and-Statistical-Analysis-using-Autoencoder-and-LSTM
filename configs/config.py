"""
Global configuration for the project.
"""

from pathlib import Path

# ==========================
# Project Paths
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "datasets" / "CMAPSSData"
OUTPUT_DIR = PROJECT_ROOT / "output"

FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"

# ==========================
# Dataset
# ==========================

DATASET = "FD001"

TRAIN_FILE = DATASET_DIR / f"train_{DATASET}.txt"
TEST_FILE = DATASET_DIR / f"test_{DATASET}.txt"
RUL_FILE = DATASET_DIR / f"RUL_{DATASET}.txt"

# ==========================
# Randomness
# ==========================

RANDOM_STATE = 42

# ==========================
# Visualization
# ==========================

FIG_DPI = 300
FIG_SIZE = (10, 6)