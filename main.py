"""
Main Pipeline
Edge AI-Based Fault Detection and Prediction
using NASA C-MAPSS Dataset
"""

from configs.config import (
    TRAIN_FILE,
    TEST_FILE,
    RUL_FILE,
    WINDOW_SIZE,
)

# ==========================================================
# Data
# ==========================================================

from src.data.loader import CMAPSSLoader
from src.data.preprocessing import Preprocessor
from src.data.normalizer import DataNormalizer
from src.data.splitter import EngineWiseSplitter

# ==========================================================
# Visualization
# ==========================================================

from src.visualization.eda_visualizer import EDAVisualizer

# ==========================================================
# Signal Processing
# ==========================================================

from src.signal_processing.noise_filter import NoiseFilter
from src.signal_processing.kalman_filter import KalmanFilterEstimator

# ==========================================================
# Statistical Analysis
# ==========================================================

from src.statistics.zscore_detector import ZScoreDetector

# ==========================================================
# Features
# ==========================================================

from src.features.windowing import WindowGenerator
from src.features.wavelet_features import WaveletFeatureExtractor


def main():

    print("\n" + "=" * 70)
    print("EDGE AI FAULT DETECTION PIPELINE")
    print("=" * 70)

    # ======================================================
    # STEP 1 : LOAD DATASET
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 1 : DATA LOADING")
    print("=" * 70)

    loader = CMAPSSLoader(
        TRAIN_FILE,
        TEST_FILE,
        RUL_FILE
    )

    train = loader.load_train()

    print(f"Training Shape : {train.shape}")

    # ======================================================
    # STEP 2 : EXPLORATORY DATA ANALYSIS
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 2 : EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    visualizer = EDAVisualizer(train)

    visualizer.engine_life_distribution()
    visualizer.sensor_variance()
    visualizer.correlation_heatmap()

    # ======================================================
    # STEP 3 : PREPROCESSING
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 3 : DATA PREPROCESSING")
    print("=" * 70)

    preprocessor = Preprocessor()

    train = preprocessor.remove_constant_features(
        train
    )

    train = preprocessor.add_rul(
        train
    )

    print(f"Preprocessed Shape : {train.shape}")

    # ======================================================
    # STEP 4 : ENGINE-WISE TRAIN / VALIDATION / TEST SPLIT
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 4 : ENGINE-WISE DATA SPLIT")
    print("=" * 70)

    splitter = EngineWiseSplitter(
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    )

    (
        train_df,
        validation_df,
        test_df
    ) = splitter.split(train)

    # ======================================================
    # STEP 5 : NORMALIZATION
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 5 : FEATURE NORMALIZATION")
    print("=" * 70)

    normalizer = DataNormalizer()

    # IMPORTANT:
    # Fit ONLY on training engines.
    normalizer.fit(train_df)

    train_df = normalizer.transform(
        train_df
    )

    validation_df = normalizer.transform(
        validation_df
    )

    test_df = normalizer.transform(
        test_df
    )

    normalizer.save(
        "output/models/scaler.pkl"
    )

    print("\n========== NORMALIZED TRAIN DATA ==========")
    print(train_df.head())

    # ======================================================
    # STEP 6 : NOISE FILTERING
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 6 : NOISE FILTERING")
    print("=" * 70)

    noise_filter = NoiseFilter()

    train_df = noise_filter.apply(
        train_df
    )

    validation_df = noise_filter.apply(
        validation_df
    )

    test_df = noise_filter.apply(
        test_df
    )

    print("\n✓ Noise filtering completed for all splits.")

    # ======================================================
    # STEP 7 : Z-SCORE STATISTICAL ANALYSIS
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 7 : Z-SCORE STATISTICAL ANALYSIS")
    print("=" * 70)

    # Each split is analyzed independently.
    # This prevents information from validation/test engines
    # influencing the statistical anomaly analysis.

    zscore_train = ZScoreDetector(
        threshold=3.0
    )

    zscore_validation = ZScoreDetector(
        threshold=3.0
    )

    zscore_test = ZScoreDetector(
        threshold=3.0
    )

    train_df = zscore_train.detect(
        train_df
    )

    validation_df = zscore_validation.detect(
        validation_df
    )

    test_df = zscore_test.detect(
        test_df
    )

    print("\n✓ Z-Score analysis completed for all splits.")

    # ======================================================
    # STEP 8 : KALMAN FILTER
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 8 : KALMAN FILTER STATE ESTIMATION")
    print("=" * 70)

    kalman_train = KalmanFilterEstimator()
    kalman_validation = KalmanFilterEstimator()
    kalman_test = KalmanFilterEstimator()

    train_df = kalman_train.apply(
        train_df
    )

    validation_df = kalman_validation.apply(
        validation_df
    )

    test_df = kalman_test.apply(
        test_df
    )

    print("\n✓ Kalman filtering completed for all splits.")

    # ======================================================
    # STEP 9 : SLIDING WINDOW GENERATION
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 9 : SLIDING WINDOW GENERATION")
    print("=" * 70)

    window_generator = WindowGenerator(
        window_size=WINDOW_SIZE,
        stride=1
    )

    # ------------------------------------------------------
    # Training windows
    # ------------------------------------------------------

    print("\n---------- TRAINING WINDOWS ----------")

    X_train, y_train, metadata_train = (
        window_generator.generate(
            train_df
        )
    )

    # ------------------------------------------------------
    # Validation windows
    # ------------------------------------------------------

    print("\n---------- VALIDATION WINDOWS ----------")

    X_validation, y_validation, metadata_validation = (
        window_generator.generate(
            validation_df
        )
    )

    # ------------------------------------------------------
    # Test windows
    # ------------------------------------------------------

    print("\n---------- TEST WINDOWS ----------")

    X_test, y_test, metadata_test = (
        window_generator.generate(
            test_df
        )
    )

    print("\n========== WINDOW SUMMARY ==========")

    print(
        f"X_train      : {X_train.shape}"
    )

    print(
        f"X_validation : {X_validation.shape}"
    )

    print(
        f"X_test       : {X_test.shape}"
    )

    print(
        f"y_train      : {y_train.shape}"
    )

    print(
        f"y_validation : {y_validation.shape}"
    )

    print(
        f"y_test       : {y_test.shape}"
    )

    # ======================================================
    # STEP 10 : WAVELET FEATURE EXTRACTION
    # ======================================================

    print("\n" + "=" * 70)
    print("STEP 10 : WAVELET FEATURE EXTRACTION")
    print("=" * 70)

    sensor_columns = [
        col
        for col in train_df.columns
        if col.startswith("sensor_")
    ]

    # ------------------------------------------------------
    # Training Wavelet Features
    # ------------------------------------------------------

    print("\n---------- TRAINING WAVELET FEATURES ----------")

    wavelet_train_extractor = (
        WaveletFeatureExtractor()
    )

    wavelet_train_extractor.set_input(
        X_train
    )

    wavelet_train = (
        wavelet_train_extractor.extract(
            X_train,
            sensor_columns
        )
    )

    # ------------------------------------------------------
    # Validation Wavelet Features
    # ------------------------------------------------------

    print("\n---------- VALIDATION WAVELET FEATURES ----------")

    wavelet_validation_extractor = (
        WaveletFeatureExtractor()
    )

    wavelet_validation_extractor.set_input(
        X_validation
    )

    wavelet_validation = (
        wavelet_validation_extractor.extract(
            X_validation,
            sensor_columns
        )
    )

    # ------------------------------------------------------
    # Test Wavelet Features
    # ------------------------------------------------------

    print("\n---------- TEST WAVELET FEATURES ----------")

    wavelet_test_extractor = (
        WaveletFeatureExtractor()
    )

    wavelet_test_extractor.set_input(
        X_test
    )

    wavelet_test = (
        wavelet_test_extractor.extract(
            X_test,
            sensor_columns
        )
    )

    # ======================================================
    # FINAL PIPELINE SUMMARY
    # ======================================================

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    print(
        f"Training Engines      : "
        f"{train_df['engine_id'].nunique()}"
    )

    print(
        f"Validation Engines    : "
        f"{validation_df['engine_id'].nunique()}"
    )

    print(
        f"Test Engines          : "
        f"{test_df['engine_id'].nunique()}"
    )

    print()

    print(
        f"Training Windows      : "
        f"{X_train.shape}"
    )

    print(
        f"Validation Windows    : "
        f"{X_validation.shape}"
    )

    print(
        f"Test Windows          : "
        f"{X_test.shape}"
    )

    print()

    print(
        f"Training Wavelet      : "
        f"{wavelet_train.shape}"
    )

    print(
        f"Validation Wavelet    : "
        f"{wavelet_validation.shape}"
    )

    print(
        f"Test Wavelet          : "
        f"{wavelet_test.shape}"
    )

    print("\n" + "=" * 70)
    print("PREPROCESSING + FEATURE ENGINEERING COMPLETED")
    print("=" * 70)

    print("\nNext Module:")
    print("→ Isolation Forest Anomaly Detection")


if __name__ == "__main__":
    main()