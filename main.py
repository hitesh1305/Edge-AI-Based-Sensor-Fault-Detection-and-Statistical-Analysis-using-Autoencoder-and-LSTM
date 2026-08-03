"""
Main Pipeline
Edge AI-Based Fault Detection and Prediction
using NASA C-MAPSS Dataset
"""

from configs.config import TRAIN_FILE, TEST_FILE, RUL_FILE

# ==========================
# Data
# ==========================
from src.data.loader import CMAPSSLoader
from src.data.preprocessing import Preprocessor
from src.data.normalizer import DataNormalizer

# ==========================
# Visualization
# ==========================
from src.visualization.eda_visualizer import EDAVisualizer

# ==========================
# Signal Processing
# ==========================
from src.signal_processing.noise_filter import NoiseFilter

# ==========================
# Statistical Analysis
# ==========================
from src.statistics.zscore_detector import ZScoreDetector


# ==========================
# Kalman Filter
# ==========================
from src.signal_processing.kalman_filter import KalmanFilterEstimator

def main():

    print("\n" + "=" * 70)
    print("EDGE AI FAULT DETECTION PIPELINE")
    print("=" * 70)

    # ==========================================================
    # STEP 1 : Load Dataset
    # ==========================================================

    loader = CMAPSSLoader(
        TRAIN_FILE,
        TEST_FILE,
        RUL_FILE
    )

    train = loader.load_train()

    # ==========================================================
    # STEP 2 : Exploratory Data Analysis
    # ==========================================================

    visualizer = EDAVisualizer(train)

    visualizer.engine_life_distribution()
    visualizer.sensor_variance()
    visualizer.correlation_heatmap()

    # ==========================================================
    # STEP 3 : Data Preprocessing
    # ==========================================================

    preprocessor = Preprocessor()

    train = preprocessor.remove_constant_features(train)

    train = preprocessor.add_rul(train)

    # ==========================================================
    # STEP 4 : Feature Normalization
    # ==========================================================

    normalizer = DataNormalizer()

    normalizer.fit(train)

    train = normalizer.transform(train)
    
    normalizer.save("output/models/scaler.pkl")
    
    print("\n========== NORMALIZED DATASET ==========")
    print(train.head())

    # ==========================================================
    # STEP 5 : Noise Filtering
    # ==========================================================

    noise_filter = NoiseFilter()

    train = noise_filter.apply(train)

    print("\n========== FILTERED DATASET ==========")
    print(train.head())

    # ==========================================================
    # STEP 6 : Z-Score Statistical Analysis
    # ==========================================================

    zscore = ZScoreDetector(threshold=3.0)

    train = zscore.detect(train)

    print("\n========== DATA AFTER Z-SCORE ==========")
    print(train.head())


    # ==========================================================
    # STEP 7 : Kalman Filter
    # ==========================================================
    
    kalman = KalmanFilterEstimator()
    
    train = kalman.apply(train)
    
    print("\n========== KALMAN FILTERED DATA ==========")
    print(train.head())
    # ==========================================================
    # Current Dataset Summary
    # ==========================================================

    print("\n" + "=" * 70)
    print("CURRENT DATASET STATUS")
    print("=" * 70)

    print(f"Rows    : {train.shape[0]}")
    print(f"Columns : {train.shape[1]}")

    print("\nLast 10 Columns:")

    print(train.columns[-10:].tolist())

    print("\nPipeline Completed Successfully up to Z-Score Analysis.")

    print("\nNext Module:")
    print("→ Extended Kalman Filter (Predictive Filtering)")


if __name__ == "__main__":
    main()