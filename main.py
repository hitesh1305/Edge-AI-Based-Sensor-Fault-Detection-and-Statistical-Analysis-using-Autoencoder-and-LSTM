"""
Main Pipeline
Edge AI-Based Fault Detection and Prediction
using NASA C-MAPSS Dataset
"""

import matplotlib

matplotlib.use("Agg")

import numpy as np

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

from src.models.isolation_forest import IsolationForestDetector

from src.models.autoencoder import AutoencoderDetector

from src.models.fault_fusion import FaultScoreFusion

from src.models.lstm_predictor import LSTMPredictor

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
        train_df,
        split_name="train"
    )

    validation_df = kalman_validation.apply(
        validation_df,
        split_name="validation"
    )

    test_df = kalman_test.apply(
        test_df,
        split_name="test"
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
            train_df,
            split_name="train"
        )
    )

    # ------------------------------------------------------
    # Validation windows
    # ------------------------------------------------------

    print("\n---------- VALIDATION WINDOWS ----------")

    X_validation, y_validation, metadata_validation = (
        window_generator.generate(
            validation_df,
            split_name="validation"
        )
    )

    # ------------------------------------------------------
    # Test windows
    # ------------------------------------------------------

    print("\n---------- TEST WINDOWS ----------")

    X_test, y_test, metadata_test = (
        window_generator.generate(
            test_df,
            split_name="test"
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
            sensor_columns,
            split_name="train"
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
            sensor_columns,
            split_name="validation"
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
            sensor_columns,
            split_name="test"
        )
    )

    # ==========================================================
    # STEP 11 : ISOLATION FOREST ANOMALY DETECTION
    # ==========================================================

    print("\n" + "=" * 70)
    print("STEP 11 : ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 70)

    isolation_forest = IsolationForestDetector(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        random_state=42,
        n_jobs=-1
    )

    # ----------------------------------------------------------
    # TRAIN
    # ----------------------------------------------------------

    training_time = isolation_forest.fit(
        wavelet_train
    )

    # ----------------------------------------------------------
    # Validation scores
    # ----------------------------------------------------------

    validation_scores = (
        isolation_forest.anomaly_scores(
            wavelet_validation
        )
    )

    validation_labels = (
        isolation_forest.create_rul_labels(
            metadata_validation,
            rul_threshold=30
        )
    )

    isolation_forest.select_threshold(
        validation_scores,
        validation_labels
    )

    validation_predictions = (
        validation_scores
        >= isolation_forest.threshold
    ).astype(int)

    validation_results = (
        isolation_forest.evaluate(
            validation_labels,
            validation_predictions,
            validation_scores,
            "validation"
        )
    )

    isolation_forest.save_results(
        validation_results
    )

    # ----------------------------------------------------------
    # Test
    # ----------------------------------------------------------

    test_predictions, test_scores = (
        isolation_forest.predict(
            wavelet_test
        )
    )

    test_labels = (
        isolation_forest.create_rul_labels(
            metadata_test,
            rul_threshold=30
        )
    )

    test_results = (
        isolation_forest.evaluate(
            test_labels,
            test_predictions,
            test_scores,
            "test"
        )
    )

    isolation_forest.save_results(
        test_results
    )

    # ----------------------------------------------------------
    # Figures
    # ----------------------------------------------------------

    isolation_forest.plot_confusion_matrix(
        test_labels,
        test_predictions,
        "test"
    )

    isolation_forest.plot_roc_curve(
        test_labels,
        test_scores,
        "test"
    )

    isolation_forest.plot_score_distribution(
        test_scores,
        test_labels,
        "test",
        isolation_forest.threshold
    )

    # ----------------------------------------------------------
    # Inference
    # ----------------------------------------------------------

    inference_results = (
        isolation_forest.measure_inference_time(
            wavelet_test
        )
    )

    print("\n✓ Isolation Forest completed.")
    
    # ==========================================================
    # STEP 12 : DENSE AUTOENCODER
    # ==========================================================

    print("\n" + "=" * 70)
    print("STEP 12 : DENSE AUTOENCODER ANOMALY DETECTION")
    print("=" * 70)

    # ----------------------------------------------------------
    # Evaluation labels
    # ----------------------------------------------------------

    train_labels = (
        metadata_train["target_rul"]
        <= 30
    ).astype(int).to_numpy()

    validation_labels = (
        metadata_validation["target_rul"]
        <= 30
    ).astype(int).to_numpy()

    test_labels = (
        metadata_test["target_rul"]
        <= 30
    ).astype(int).to_numpy()

    # ----------------------------------------------------------
    # Flatten Wavelet Features
    # ----------------------------------------------------------

    X_wavelet_train = (
        wavelet_train
        .drop(columns=["window_id"])
        .to_numpy(dtype=np.float32)
    )

    X_wavelet_validation = (
        wavelet_validation
        .drop(columns=["window_id"])
        .to_numpy(dtype=np.float32)
    )

    X_wavelet_test = (
        wavelet_test
        .drop(columns=["window_id"])
        .to_numpy(dtype=np.float32)
    )

    # ----------------------------------------------------------
    # Train ONLY on healthy training samples
    # ----------------------------------------------------------

    healthy_train_mask = (
        train_labels == 0
    )

    healthy_validation_mask = (
        validation_labels == 0
    )

    X_wavelet_train_healthy = (
        X_wavelet_train[
            healthy_train_mask
        ]
    )

    X_wavelet_validation_healthy = (
        X_wavelet_validation[
            healthy_validation_mask
        ]
    )

    print(
        f"\nHealthy Training Samples   : "
        f"{len(X_wavelet_train_healthy)}"
    )

    print(
        f"Healthy Validation Samples : "
        f"{len(X_wavelet_validation_healthy)}"
    )

    # ----------------------------------------------------------
    # Create Autoencoder
    # ----------------------------------------------------------

    autoencoder = AutoencoderDetector(
        input_dim=X_wavelet_train.shape[1],
        encoding_dim=64,
        learning_rate=0.001,
        batch_size=128,
        epochs=50
    )

    # ----------------------------------------------------------
    # Train
    # ----------------------------------------------------------

    history, autoencoder_training_time = (
        autoencoder.fit(
            X_wavelet_train_healthy,
            X_wavelet_validation_healthy
        )
    )

    # ----------------------------------------------------------
    # Training History
    # ----------------------------------------------------------

    autoencoder.plot_training_history(
        history
    )

    # ----------------------------------------------------------
    # Validation Reconstruction Error
    # ----------------------------------------------------------

    validation_errors = (
        autoencoder.reconstruction_errors(
            X_wavelet_validation
        )
    )

    # ----------------------------------------------------------
    # Select Threshold ONLY on Validation
    # ----------------------------------------------------------

    autoencoder.select_threshold(
        validation_errors,
        validation_labels
    )

    validation_predictions = (
        validation_errors
        >= autoencoder.threshold
    ).astype(int)

    validation_results = (
        autoencoder.evaluate(
            validation_labels,
            validation_predictions,
            validation_errors,
            "validation"
        )
    )

    autoencoder.save_results(
        validation_results
    )

    # ----------------------------------------------------------
    # Test
    # ----------------------------------------------------------

    test_predictions, test_errors = (
        autoencoder.predict(
            X_wavelet_test
        )
    )

    test_results = (
        autoencoder.evaluate(
            test_labels,
            test_predictions,
            test_errors,
            "test"
        )
    )

    autoencoder.save_results(
        test_results
    )

    # ----------------------------------------------------------
    # Required Paper Figures
    # ----------------------------------------------------------

    autoencoder.plot_error_distribution(
        test_errors,
        test_labels,
        autoencoder.threshold,
        "test"
    )

    autoencoder.plot_confusion_matrix(
        test_labels,
        test_predictions,
        "test"
    )

    autoencoder.plot_roc_curve(
        test_labels,
        test_errors,
        "test"
    )

    # ----------------------------------------------------------
    # Inference Time
    # ----------------------------------------------------------

    autoencoder_inference = (
        autoencoder.measure_inference_time(
            X_wavelet_test
        )
    )

    # ----------------------------------------------------------
    # Save Model
    # ----------------------------------------------------------

    autoencoder.save_model()

    print("\n✓ Dense Autoencoder completed.")


    # ==========================================================
    # STEP 13 : PROPOSED FAULT SCORE FUSION
    # ==========================================================

    print("\n" + "=" * 70)
    print("STEP 13 : PROPOSED FAULT SCORE FUSION")
    print("=" * 70)

    """
    Research Design
    ---------------
    Isolation Forest and Dense Autoencoder provide complementary
    anomaly evidence.

    Isolation Forest:
        Detects deviations in the wavelet feature space.

    Dense Autoencoder:
        Detects abnormal reconstruction behavior.

    Their continuous anomaly scores are normalized using
    VALIDATION data only and combined using a 50:50 weighted
    fusion.

    The fusion threshold is selected using VALIDATION F1.

    The TEST set is used only for final evaluation.
    """

    # ----------------------------------------------------------
    # Create Fusion Model
    # ----------------------------------------------------------

    fault_fusion = FaultScoreFusion(
        isolation_weight=0.5,
        autoencoder_weight=0.5
    )

    print("\nFusion Weights")
    print(
        f"Isolation Forest : "
        f"{fault_fusion.isolation_weight:.2f}"
    )

    print(
        f"Autoencoder      : "
        f"{fault_fusion.autoencoder_weight:.2f}"
    )

    # ----------------------------------------------------------
    # VALIDATION SCORE NORMALIZATION
    # ----------------------------------------------------------
    #
    # IMPORTANT:
    # The normalization parameters are learned ONLY from
    # validation data.
    #
    # Test information is not used during threshold selection.

    fault_fusion.fit_scaler(
        validation_scores,
        validation_errors
    )

    (
        validation_if_normalized,
        validation_ae_normalized
    ) = fault_fusion.transform_scores(
        validation_scores,
        validation_errors
    )

    # ----------------------------------------------------------
    # VALIDATION FUSION SCORE
    # ----------------------------------------------------------

    validation_fused_scores = (
        fault_fusion.fuse_scores(
            validation_if_normalized,
            validation_ae_normalized
        )
    )

    # ----------------------------------------------------------
    # SELECT FUSION THRESHOLD
    # ----------------------------------------------------------
    #
    # Threshold is selected ONLY using validation data.

    fault_fusion.select_threshold(
        validation_fused_scores,
        validation_labels
    )

    validation_fusion_predictions = (
        fault_fusion.predict(
            validation_fused_scores
        )
    )

    # ----------------------------------------------------------
    # VALIDATION EVALUATION
    # ----------------------------------------------------------

    fusion_validation_results = (
        fault_fusion.evaluate(
            validation_labels,
            validation_fusion_predictions,
            validation_fused_scores,
            "validation"
        )
    )

    # ----------------------------------------------------------
    # TEST SCORE NORMALIZATION
    # ----------------------------------------------------------
    #
    # The scaler fitted on validation data is reused here.
    # No fitting is performed on test data.

    (
        test_if_normalized,
        test_ae_normalized
    ) = fault_fusion.transform_scores(
        test_scores,
        test_errors
    )

    # ----------------------------------------------------------
    # TEST FUSION SCORE
    # ----------------------------------------------------------

    test_fused_scores = (
        fault_fusion.fuse_scores(
            test_if_normalized,
            test_ae_normalized
        )
    )

    # ----------------------------------------------------------
    # FINAL TEST PREDICTION
    # ----------------------------------------------------------
    #
    # The threshold was determined on validation data.
    # It is NOT recalculated using the test set.

    test_fusion_predictions = (
        fault_fusion.predict(
            test_fused_scores
        )
    )

    # ----------------------------------------------------------
    # FINAL TEST EVALUATION
    # ----------------------------------------------------------

    fusion_test_results = (
        fault_fusion.evaluate(
            test_labels,
            test_fusion_predictions,
            test_fused_scores,
            "test"
        )
    )

    # ----------------------------------------------------------
    # SAVE RAW + NORMALIZED FUSION SCORES
    # ----------------------------------------------------------

    fault_fusion.save_scores(
        window_ids=metadata_test["window_id"].to_numpy(),
        labels=test_labels,
        isolation_scores=test_scores,
        autoencoder_errors=test_errors,
        isolation_normalized=test_if_normalized,
        autoencoder_normalized=test_ae_normalized,
        fused_scores=test_fused_scores,
        predictions=test_fusion_predictions,
        output_path=(
            "output/tables/fusion/"
            "test_anomaly_scores.csv"
        )
    )

    # ----------------------------------------------------------
    # SAVE VALIDATION SCORES
    # ----------------------------------------------------------

    fault_fusion.save_scores(
        window_ids=metadata_validation["window_id"].to_numpy(),
        labels=validation_labels,
        isolation_scores=validation_scores,
        autoencoder_errors=validation_errors,
        isolation_normalized=validation_if_normalized,
        autoencoder_normalized=validation_ae_normalized,
        fused_scores=validation_fused_scores,
        predictions=validation_fusion_predictions,
        output_path=(
            "output/tables/fusion/"
            "validation_anomaly_scores.csv"
        )
    )

    # ----------------------------------------------------------
    # SAVE FINAL TEST RESULTS
    # ----------------------------------------------------------

    fault_fusion.save_results(
        fusion_test_results,
        "output/tables/fusion/results.csv"
    )

    # ----------------------------------------------------------
    # CONFUSION MATRIX
    # ----------------------------------------------------------

    fault_fusion.plot_confusion_matrix(
        test_labels,
        test_fusion_predictions,
        "output/figures/fusion_confusion_test.png"
    )

    # ----------------------------------------------------------
    # ROC CURVE
    # ----------------------------------------------------------

    fault_fusion.plot_roc_curve(
        test_labels,
        test_fused_scores,
        "output/figures/fusion_roc_test.png"
    )

    # ----------------------------------------------------------
    # SCORE DISTRIBUTION
    # ----------------------------------------------------------

    fault_fusion.plot_score_distribution(
        test_fused_scores,
        test_labels,
        fault_fusion.threshold,
        "output/figures/fusion_score_distribution_test.png"
    )

    # ----------------------------------------------------------
    # INFERENCE PERFORMANCE
    # ----------------------------------------------------------

    fusion_inference = (
        fault_fusion.measure_inference_time(
            test_scores,
            test_errors
        )
    )

    print("\n✓ Proposed fault score fusion completed.")
    # ==========================================================
    # STEP 14 : LSTM RUL PREDICTION
    # ==========================================================

    print("\n" + "=" * 70)
    print("STEP 14 : LSTM-BASED RUL PREDICTION")
    print("=" * 70)

    # ----------------------------------------------------------
    # Prepare LSTM input
    # ----------------------------------------------------------
    # The LSTM uses the original temporal sensor windows.
    #
    # Shape:
    # Training   : (12377, 30, 17)
    # Validation : (2725, 30, 17)
    # Test       : (2629, 30, 17)

    X_train_lstm = X_train.astype(np.float32)
    X_validation_lstm = X_validation.astype(np.float32)
    X_test_lstm = X_test.astype(np.float32)

    # ----------------------------------------------------------
    # RUL targets
    # ----------------------------------------------------------
    # The target RUL corresponds to the END of each window.
    #
    # IMPORTANT:
    # We use metadata generated by WindowGenerator rather than
    # reconstructing RUL values manually.

    y_train_lstm = (
        metadata_train["target_rul"]
        .to_numpy(dtype=np.float32)
    )

    y_validation_lstm = (
        metadata_validation["target_rul"]
        .to_numpy(dtype=np.float32)
    )

    y_test_lstm = (
        metadata_test["target_rul"]
        .to_numpy(dtype=np.float32)
    )

    print("\n" + "-" * 70)
    print("LSTM RUL DATA")
    print("-" * 70)

    print(
        "Training Input Shape   :",
        X_train_lstm.shape
    )

    print(
        "Validation Input Shape :",
        X_validation_lstm.shape
    )

    print(
        "Test Input Shape       :",
        X_test_lstm.shape
    )

    print(
        "Training RUL Shape      :",
        y_train_lstm.shape
    )

    print(
        "Validation RUL Shape   :",
        y_validation_lstm.shape
    )

    print(
        "Test RUL Shape         :",
        y_test_lstm.shape
    )

    print(
        "\nTraining RUL Range     :",
        f"{y_train_lstm.min():.0f} - "
        f"{y_train_lstm.max():.0f}"
    )

    print(
        "Validation RUL Range   :",
        f"{y_validation_lstm.min():.0f} - "
        f"{y_validation_lstm.max():.0f}"
    )

    print(
        "Test RUL Range         :",
        f"{y_test_lstm.min():.0f} - "
        f"{y_test_lstm.max():.0f}"
    )

    # ----------------------------------------------------------
    # Create LSTM Predictor
    # ----------------------------------------------------------

    lstm_predictor = LSTMPredictor(
        timesteps=X_train_lstm.shape[1],
        features=X_train_lstm.shape[2],
        lstm_units=64,
        learning_rate=0.001,
        batch_size=128,
        epochs=50,
        dropout=0.2,
    )

    # ----------------------------------------------------------
    # Train LSTM
    # ----------------------------------------------------------

    lstm_history, lstm_training_time = (
        lstm_predictor.fit(
            X_train_lstm,
            y_train_lstm,
            X_validation_lstm,
            y_validation_lstm,
        )
    )

    # ----------------------------------------------------------
    # Training History
    # ----------------------------------------------------------

    lstm_predictor.plot_training_history(
        lstm_history
    )

    # ----------------------------------------------------------
    # Validation Prediction
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("LSTM VALIDATION RUL PREDICTION")
    print("=" * 70)

    validation_predictions_rul = (
        lstm_predictor.predict(
            X_validation_lstm
        )
    )

    validation_rul_results = (
        lstm_predictor.evaluate(
            y_validation_lstm,
            validation_predictions_rul,
            "validation",
        )
    )

    lstm_predictor.save_results(
        validation_rul_results
    )

    # ----------------------------------------------------------
    # Validation Figures
    # ----------------------------------------------------------

    lstm_predictor.plot_predictions(
        y_validation_lstm,
        validation_predictions_rul,
        "validation",
    )

    lstm_predictor.plot_error_distribution(
        y_validation_lstm,
        validation_predictions_rul,
        "validation",
    )

    lstm_predictor.plot_actual_vs_predicted(
        y_validation_lstm,
        validation_predictions_rul,
        "validation",
    )

    # ----------------------------------------------------------
    # FINAL TEST PREDICTION
    # ----------------------------------------------------------

    print("\n" + "=" * 70)
    print("LSTM FINAL TEST RUL PREDICTION")
    print("=" * 70)

    test_predictions_rul = (
        lstm_predictor.predict(
            X_test_lstm
        )
    )

    test_rul_results = (
        lstm_predictor.evaluate(
            y_test_lstm,
            test_predictions_rul,
            "test",
        )
    )

    lstm_predictor.save_results(
        test_rul_results
    )

    # ----------------------------------------------------------
    # Test Figures
    # ----------------------------------------------------------

    lstm_predictor.plot_predictions(
        y_test_lstm,
        test_predictions_rul,
        "test",
    )

    lstm_predictor.plot_error_distribution(
        y_test_lstm,
        test_predictions_rul,
        "test",
    )

    lstm_predictor.plot_actual_vs_predicted(
        y_test_lstm,
        test_predictions_rul,
        "test",
    )

    # ----------------------------------------------------------
    # Inference Performance
    # ----------------------------------------------------------

    lstm_inference_results = (
        lstm_predictor.measure_inference_time(
            X_test_lstm
        )
    )

    # ----------------------------------------------------------
    # Save Model
    # ----------------------------------------------------------

    lstm_predictor.save_model()

    print(
        "\n✓ LSTM RUL Prediction completed."
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