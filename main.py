"""
Main Pipeline
Edge AI-Based Fault Detection and Prediction
using NASA C-MAPSS Dataset
"""

import matplotlib

matplotlib.use("Agg")
from pathlib import Path
import numpy as np
import pandas as pd


from configs.config import (
    TRAIN_FILE,
    TEST_FILE,
    RUL_FILE,
    WINDOW_SIZE,
)

from sklearn.preprocessing import StandardScaler
import joblib

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
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
    # STEP 12 : DENSE AUTOENCODER ANOMALY DETECTION
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
    # Healthy sample masks
    # ----------------------------------------------------------

    healthy_train_mask = (
        train_labels == 0
    )

    healthy_validation_mask = (
        validation_labels == 0
    )

    print(
        f"\nTotal Training Windows      : "
        f"{len(X_wavelet_train)}"
    )

    print(
        f"Healthy Training Windows    : "
        f"{np.sum(healthy_train_mask)}"
    )

    print(
        f"Total Validation Windows    : "
        f"{len(X_wavelet_validation)}"
    )

    print(
        f"Healthy Validation Windows  : "
        f"{np.sum(healthy_validation_mask)}"
    )

    print(
        f"Total Test Windows          : "
        f"{len(X_wavelet_test)}"
    )

    # ----------------------------------------------------------
    # Extract healthy training samples
    # ----------------------------------------------------------

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

    # ----------------------------------------------------------
    # Wavelet Feature Normalization
    # ----------------------------------------------------------
    #
    # IMPORTANT:
    # The scaler is fitted ONLY on healthy training data.
    #
    # Validation and test data are transformed using the
    # training-fitted scaler.
    #
    # This prevents test information from influencing training.
    # ----------------------------------------------------------

    

    wavelet_scaler = StandardScaler()

    wavelet_scaler.fit(
        X_wavelet_train_healthy
    )

    X_wavelet_train_scaled = (
        wavelet_scaler.transform(
            X_wavelet_train
        )
    )

    X_wavelet_validation_scaled = (
        wavelet_scaler.transform(
            X_wavelet_validation
        )
    )

    X_wavelet_test_scaled = (
        wavelet_scaler.transform(
            X_wavelet_test
        )
    )

    joblib.dump(
        wavelet_scaler,
        "output/models/wavelet_scaler.pkl"
    )

    print(
        "\n✓ Wavelet scaler fitted on "
        "healthy training data."
    )

    print(
        "✓ Wavelet scaler saved to "
        "output/models/wavelet_scaler.pkl"
    )

    # ----------------------------------------------------------
    # Select healthy scaled samples for Autoencoder training
    # ----------------------------------------------------------

    X_wavelet_train_healthy = (
        X_wavelet_train_scaled[
            healthy_train_mask
        ]
    )

    X_wavelet_validation_healthy = (
        X_wavelet_validation_scaled[
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
        input_dim=X_wavelet_train_scaled.shape[1],
        encoding_dim=64,
        learning_rate=0.0005,
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
            X_wavelet_validation_scaled
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
            X_wavelet_test_scaled
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
            X_wavelet_test_scaled
        )
    )

    # ----------------------------------------------------------
    # Save Model
    # ----------------------------------------------------------

    autoencoder.save_model()

    print(
        "\n✓ Dense Autoencoder completed."
    )
    # ==========================================================
    # STEP 13 : PROPOSED FAULT SCORE FUSION
    # ==========================================================
    
    print("\n" + "=" * 70)
    print("STEP 13 : PROPOSED FAULT SCORE FUSION")
    print("=" * 70)
    
    """
    Research Design
    ---------------
    
    The proposed fault detector combines two complementary anomaly
    indicators:
    
    1. Isolation Forest anomaly score
    2. Dense Autoencoder reconstruction error
    
    Isolation Forest:
        Detects deviations in the wavelet-feature space.
    
    Dense Autoencoder:
        Detects abnormal reconstruction behavior.
    
    Both scores are first aligned so that:
    
        Higher score = higher probability of fault/anomaly
    
    The score normalization parameters are learned using VALIDATION
    data only.
    
    The normalized scores are then combined using a weighted fusion:
    
        Fused Score =
            w_IF * IF_Score
            +
            w_AE * AE_Score
    
    The fusion threshold is selected using validation F1.
    
    The TEST set is used only once for final evaluation.
    
    IMPORTANT:
        Isolation Forest score direction is determined from the
        validation ROC-AUC rather than blindly applying (1-score).
    """
    
    
    # ==========================================================
    # 1. CREATE FUSION MODEL
    # ==========================================================
    
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
    
    
    # ==========================================================
    # 2. CONVERT INPUTS TO NUMPY ARRAYS
    # ==========================================================
    
    validation_if_scores = np.asarray(
        validation_scores,
        dtype=np.float64
    ).reshape(-1)
    
    validation_ae_errors = np.asarray(
        validation_errors,
        dtype=np.float64
    ).reshape(-1)
    
    validation_labels_array = np.asarray(
        validation_labels,
        dtype=int
    ).reshape(-1)
    
    test_if_scores = np.asarray(
        test_scores,
        dtype=np.float64
    ).reshape(-1)
    
    test_ae_errors = np.asarray(
        test_errors,
        dtype=np.float64
    ).reshape(-1)
    
    test_labels_array = np.asarray(
        test_labels,
        dtype=int
    ).reshape(-1)
    
    
    # ==========================================================
    # 3. CHECK SCORE LENGTHS
    # ==========================================================
    
    if not (
        len(validation_if_scores)
        == len(validation_ae_errors)
        == len(validation_labels_array)
    ):
    
        raise ValueError(
            "Validation Isolation Forest scores, "
            "Autoencoder errors and labels must have "
            "the same number of samples."
        )
    
    
    if not (
        len(test_if_scores)
        == len(test_ae_errors)
        == len(test_labels_array)
    ):
    
        raise ValueError(
            "Test Isolation Forest scores, "
            "Autoencoder errors and labels must have "
            "the same number of samples."
        )
    
    
    # ==========================================================
    # 4. DETERMINE ISOLATION FOREST SCORE DIRECTION
    # ==========================================================
    #
    # We do NOT assume the direction.
    #
    # The validation labels are used only to determine whether
    # higher or lower Isolation Forest scores correspond to faults.
    #
    # This is legitimate calibration because validation data is
    # explicitly reserved for model calibration and threshold
    # selection.
    #
    # If:
    #
    #     AUC(raw score) > AUC(-raw score)
    #
    # then higher raw score = more anomalous.
    #
    # Otherwise:
    #
    #     higher anomaly score = -raw score
    #
    # ==========================================================
    
    try:
    
        if_auc_raw = roc_auc_score(
            validation_labels_array,
            validation_if_scores
        )
    
        if_auc_reversed = roc_auc_score(
            validation_labels_array,
            -validation_if_scores
        )
    
    except ValueError:
    
        raise ValueError(
            "Validation labels must contain both normal and "
            "fault samples for score-direction calibration."
        )
    
    
    if if_auc_reversed > if_auc_raw:
    
        isolation_direction = -1
    
        validation_if_aligned = (
            -validation_if_scores
        )
    
        direction_text = (
            "Reversed (higher anomaly score = -raw IF score)"
        )
    
    else:
    
        isolation_direction = 1
    
        validation_if_aligned = (
            validation_if_scores
        )
    
        direction_text = (
            "Raw (higher raw IF score = higher anomaly)"
        )
    
    
    print("\n" + "=" * 70)
    print("ISOLATION FOREST SCORE DIRECTION CALIBRATION")
    print("=" * 70)
    
    print(
        f"Validation ROC-AUC (raw)      : "
        f"{if_auc_raw:.4f}"
    )
    
    print(
        f"Validation ROC-AUC (reversed) : "
        f"{if_auc_reversed:.4f}"
    )
    
    print(
        f"Selected Direction            : "
        f"{direction_text}"
    )
    
    
    # ==========================================================
    # 5. ALIGN TEST ISOLATION FOREST SCORES
    # ==========================================================
    #
    # IMPORTANT:
    # The direction selected using validation data is reused
    # unchanged on the test set.
    #
    # No test labels are used here.
    # ==========================================================
    
    test_if_aligned = (
        isolation_direction
        * test_if_scores
    )
    
    
    # ==========================================================
    # 6. VALIDATION-ONLY SCORE CALIBRATION
    # ==========================================================
    #
    # The MinMaxScaler is fitted ONLY using validation scores.
    #
    # Test scores are transformed using the already fitted scaler.
    #
    # No test statistics are used to fit the scaler.
    # ==========================================================
    
    fault_fusion.fit_scaler(
        validation_if_aligned,
        validation_ae_errors
    )
    
    print("\n" + "=" * 70)
    print("FUSION SCORE CALIBRATION")
    print("=" * 70)
    
    print(
        f"Isolation Forest calibration : "
        f"{validation_if_aligned.min():.8f} "
        f"to "
        f"{validation_if_aligned.max():.8f}"
    )
    
    print(
        f"Autoencoder calibration      : "
        f"{validation_ae_errors.min():.8f} "
        f"to "
        f"{validation_ae_errors.max():.8f}"
    )
    
    print(
        "\n✓ Calibration parameters learned "
        "from validation data only."
    )
    
    
    # ==========================================================
    # 7. NORMALIZE VALIDATION SCORES
    # ==========================================================
    
    (
        validation_if_normalized,
        validation_ae_normalized
    ) = fault_fusion.transform_scores(
        validation_if_aligned,
        validation_ae_errors
    )
    
    
    # ==========================================================
    # 8. VALIDATION FUSION SCORE
    # ==========================================================
    
    validation_fused_scores = (
        fault_fusion.fuse_scores(
            validation_if_normalized,
            validation_ae_normalized
        )
    )
    
    
    # ==========================================================
    # 9. VERIFY FUSION SCORE DIRECTION
    # ==========================================================
    
    validation_fusion_auc = roc_auc_score(
        validation_labels_array,
        validation_fused_scores
    )
    
    print("\n" + "=" * 70)
    print("VALIDATION FUSION SCORE CHECK")
    print("=" * 70)
    
    print(
        f"Validation Fusion ROC-AUC : "
        f"{validation_fusion_auc:.4f}"
    )
    
    if validation_fusion_auc < 0.5:
    
        print(
            "\nWARNING: Fusion ROC-AUC is below 0.50."
        )
    
        print(
            "The anomaly-score direction appears incorrect."
        )
    
        raise RuntimeError(
            "Fusion score direction is incorrect. "
            "Check Isolation Forest and Autoencoder score definitions."
        )
    
    else:
    
        print(
            "✓ Fusion score direction is valid "
            "(higher score = greater anomaly likelihood)."
        )
    
    
    # ==========================================================
    # 10. SELECT FUSION THRESHOLD USING VALIDATION DATA
    # ==========================================================
    #
    # Threshold selection uses validation labels only.
    #
    # TEST DATA IS NOT USED.
    # ==========================================================
    
    fault_fusion.select_threshold(
        validation_fused_scores,
        validation_labels_array
    )
    
    
    # ==========================================================
    # 11. VALIDATION PREDICTIONS
    # ==========================================================
    
    validation_fusion_predictions = (
        fault_fusion.predict(
            validation_fused_scores
        )
    )
    
    
    # ==========================================================
    # 12. VALIDATION EVALUATION
    # ==========================================================
    
    fusion_validation_results = (
        fault_fusion.evaluate(
            validation_labels_array,
            validation_fusion_predictions,
            validation_fused_scores,
            "validation"
        )
    )
    
    
    # ==========================================================
    # 13. SAVE VALIDATION SCORES
    # ==========================================================
    
    fault_fusion.save_scores(
        window_ids=(
            metadata_validation["window_id"].to_numpy()
        ),
        labels=validation_labels_array,
        isolation_scores=validation_if_scores,
        autoencoder_errors=validation_ae_errors,
        isolation_normalized=validation_if_normalized,
        autoencoder_normalized=validation_ae_normalized,
        fused_scores=validation_fused_scores,
        predictions=validation_fusion_predictions,
        output_path=(
            "output/tables/fusion/"
            "validation_anomaly_scores.csv"
        )
    )
    
    
    # ==========================================================
    # 14. SAVE VALIDATION RESULTS
    # ==========================================================
    
    fault_fusion.save_results(
        fusion_validation_results,
        "output/tables/fusion/validation_results.csv"
    )
    
    
    # ==========================================================
    # 15. TRANSFORM TEST SCORES
    # ==========================================================
    #
    # IMPORTANT:
    #
    # The scaler is NOT fitted again.
    #
    # Validation calibration parameters are reused.
    # ==========================================================
    
    (
        test_if_normalized,
        test_ae_normalized
    ) = fault_fusion.transform_scores(
        test_if_aligned,
        test_ae_errors
    )
    
    
    # ==========================================================
    # 16. TEST FUSION SCORE
    # ==========================================================
    
    test_fused_scores = (
        fault_fusion.fuse_scores(
            test_if_normalized,
            test_ae_normalized
        )
    )
    
    
    # ==========================================================
    # 17. FINAL TEST PREDICTION
    # ==========================================================
    #
    # The threshold was selected using validation data.
    #
    # It is NOT recalculated using test data.
    # ==========================================================
    
    test_fusion_predictions = (
        fault_fusion.predict(
            test_fused_scores
        )
    )
    
    
    # ==========================================================
    # 18. FINAL TEST EVALUATION
    # ==========================================================
    
    fusion_test_results = (
        fault_fusion.evaluate(
            test_labels_array,
            test_fusion_predictions,
            test_fused_scores,
            "test"
        )
    )
    
    
    # ==========================================================
    # 19. SAVE TEST SCORES
    # ==========================================================
    
    fault_fusion.save_scores(
        window_ids=(
            metadata_test["window_id"].to_numpy()
        ),
        labels=test_labels_array,
        isolation_scores=test_if_scores,
        autoencoder_errors=test_ae_errors,
        isolation_normalized=test_if_normalized,
        autoencoder_normalized=test_ae_normalized,
        fused_scores=test_fused_scores,
        predictions=test_fusion_predictions,
        output_path=(
            "output/tables/fusion/"
            "test_anomaly_scores.csv"
        )
    )
    
    
    # ==========================================================
    # 20. SAVE TEST RESULTS
    # ==========================================================
    
    fault_fusion.save_results(
        fusion_test_results,
        "output/tables/fusion/results.csv"
    )
    
    
    # ==========================================================
    # 21. CONFUSION MATRIX
    # ==========================================================
    
    fault_fusion.plot_confusion_matrix(
        test_labels_array,
        test_fusion_predictions,
        "output/figures/fusion_confusion_test.png",
        title=(
            "Proposed Fault Score Fusion - "
            "Test Confusion Matrix"
        )
    )
    
    
    # ==========================================================
    # 22. ROC CURVE
    # ==========================================================
    
    fault_fusion.plot_roc_curve(
        test_labels_array,
        test_fused_scores,
        "output/figures/fusion_roc_test.png",
        title=(
            "Proposed Fault Score Fusion - "
            "Test ROC Curve"
        )
    )
    
    
    # ==========================================================
    # 23. SCORE DISTRIBUTION
    # ==========================================================
    
    fault_fusion.plot_score_distribution(
        test_fused_scores,
        test_labels_array,
        fault_fusion.threshold,
        "output/figures/fusion_score_distribution_test.png",
        title=(
            "Proposed Fault Score Fusion - "
            "Test Score Distribution"
        )
    )
    
    
    # ==========================================================
    # 24. INFERENCE PERFORMANCE
    # ==========================================================
    #
    # The inference function expects the original scores and applies
    # the validation-fitted scaler.
    #
    # Therefore we pass the direction-aligned Isolation Forest scores.
    # ==========================================================
    
    fusion_inference = (
        fault_fusion.measure_inference_time(
            test_if_aligned,
            test_ae_errors
        )
    )
    
    
    # ==========================================================
    # 25. FINAL SUMMARY
    # ==========================================================
    
    print("\n" + "=" * 70)
    print("PROPOSED FUSION SUMMARY")
    print("=" * 70)
    
    print(
        f"Isolation Forest Weight : "
        f"{fault_fusion.isolation_weight:.2f}"
    )
    
    print(
        f"Autoencoder Weight      : "
        f"{fault_fusion.autoencoder_weight:.2f}"
    )
    
    print(
        f"Isolation Forest Direction : "
        f"{direction_text}"
    )
    
    print(
        f"Fusion Threshold         : "
        f"{fault_fusion.threshold:.8f}"
    )
    
    print(
        f"Validation F1            : "
        f"{fusion_validation_results['f1_score']:.4f}"
    )
    
    print(
        f"Validation ROC-AUC       : "
        f"{fusion_validation_results['roc_auc']:.4f}"
    )
    
    print(
        f"Test Accuracy            : "
        f"{fusion_test_results['accuracy']:.4f}"
    )
    
    print(
        f"Test Precision           : "
        f"{fusion_test_results['precision']:.4f}"
    )
    
    print(
        f"Test Recall              : "
        f"{fusion_test_results['recall']:.4f}"
    )
    
    print(
        f"Test F1                  : "
        f"{fusion_test_results['f1_score']:.4f}"
    )
    
    print(
        f"Test ROC-AUC             : "
        f"{fusion_test_results['roc_auc']:.4f}"
    )
    
    print(
        "\n✓ Isolation Forest score direction "
        "calibrated using validation data."
    )
    
    print(
        "✓ Validation-only score normalization maintained."
    )
    
    print(
        "✓ Validation-only threshold selection maintained."
    )
    
    print(
        "✓ Test set used only for final evaluation."
    )
    
    print(
        "\n✓ Proposed fault score fusion completed."
    )
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


    # ==========================================================
    # STEP 15 : FINAL MODEL COMPARISON & INTEGRATED RESULTS
    # ==========================================================

    print("\n" + "=" * 70)
    print("STEP 15 : FINAL MODEL COMPARISON & INTEGRATED RESULTS")
    print("=" * 70)

    """
    Final Research Evaluation
    --------------------------

    This step ONLY consolidates the final evaluation results.

    Fault Detection:
        1. Isolation Forest
        2. Dense Autoencoder
        3. Proposed IF + Autoencoder Fusion

    RUL Prediction:
        4. LSTM RUL Predictor

    No model is retrained in this step.
    No threshold is optimized in this step.
    No test data is used for calibration.
    """

    # ==========================================================
    # 1. OUTPUT DIRECTORIES
    # ==========================================================

    final_table_dir = Path(
        "output/tables/final"
    )

    final_figure_dir = Path(
        "output/figures/final"
    )

    final_table_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    final_figure_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    # ==========================================================
    # 2. HELPER FUNCTION
    # ==========================================================

    def get_latest_test_result(
        csv_path,
        model_name
    ):

        default_result = {
            "model": model_name,
            "split": "test",
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1_score": np.nan,
            "roc_auc": np.nan
        }

        try:

            csv_path = Path(csv_path)

            if not csv_path.exists():

                print(
                    f"\nWARNING: File not found:"
                    f"\n  {csv_path}"
                )

                return default_result

            df = pd.read_csv(csv_path)

            if "split" in df.columns:

                test_rows = df[
                    df["split"]
                    .astype(str)
                    .str.lower()
                    .eq("test")
                ]

            else:

                test_rows = df

            if len(test_rows) == 0:

                print(
                    f"\nWARNING: No test result found for "
                    f"{model_name}."
                )

                return default_result

            # Latest result from the current results file
            row = test_rows.iloc[-1]

            return {
                "model": model_name,
                "split": "test",

                "accuracy": float(
                    row["accuracy"]
                ),

                "precision": float(
                    row["precision"]
                ),

                "recall": float(
                    row["recall"]
                ),

                "f1_score": float(
                    row["f1_score"]
                ),

                "roc_auc": float(
                    row["roc_auc"]
                )
            }

        except Exception as e:

            print(
                f"\nWARNING: Could not read "
                f"{model_name} results."
            )

            print(
                f"Reason: {e}"
            )

            return default_result


    # ==========================================================
    # 3. ISOLATION FOREST
    # ==========================================================

    isolation_forest_test = get_latest_test_result(
        "output/tables/isolation_forest_results.csv",
        "Isolation Forest"
    )


    # ==========================================================
    # 4. AUTOENCODER
    # ==========================================================

    autoencoder_test = get_latest_test_result(
        "output/tables/autoencoder/results.csv",
        "Dense Autoencoder"
    )


    # ==========================================================
    # 5. PROPOSED FUSION
    # ==========================================================

    proposed_fusion_test = {
        "model":
            "Proposed IF + Autoencoder Fusion",

        "split":
            "test",

        "accuracy":
            float(
                fusion_test_results["accuracy"]
            ),

        "precision":
            float(
                fusion_test_results["precision"]
            ),

        "recall":
            float(
                fusion_test_results["recall"]
            ),

        "f1_score":
            float(
                fusion_test_results["f1_score"]
            ),

        "roc_auc":
            float(
                fusion_test_results["roc_auc"]
            )
    }


    # ==========================================================
    # 6. FAULT DETECTION COMPARISON
    # ==========================================================

    fault_detection_comparison = pd.DataFrame(
        [
            isolation_forest_test,
            autoencoder_test,
            proposed_fusion_test
        ]
    )

    fault_detection_path = (
        final_table_dir
        / "fault_detection_comparison.csv"
    )

    fault_detection_comparison.to_csv(
        fault_detection_path,
        index=False
    )


    # ==========================================================
    # 7. DISPLAY FAULT DETECTION RESULTS
    # ==========================================================

    print("\n" + "-" * 70)
    print("FINAL FAULT DETECTION PERFORMANCE - TEST")
    print("-" * 70)

    display_columns = [
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]

    display_df = (
        fault_detection_comparison[
            display_columns
        ].copy()
    )

    for column in [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]:

        display_df[column] = (
            display_df[column] * 100
        )

    print(
        display_df.to_string(
            index=False,
            formatters={
                "accuracy":
                    lambda x: f"{x:.2f}%",
                "precision":
                    lambda x: f"{x:.2f}%",
                "recall":
                    lambda x: f"{x:.2f}%",
                "f1_score":
                    lambda x: f"{x:.2f}%",
                "roc_auc":
                    lambda x: f"{x:.2f}%"
            }
        )
    )

    print(
        f"\n✓ Saved:"
        f"\n  {fault_detection_path}"
    )


    # ==========================================================
    # 8. RUL PERFORMANCE
    # ==========================================================

    rul_comparison = pd.DataFrame(
        [
            {
                "model":
                    "LSTM RUL Predictor",

                "split":
                    "test",

                "mae":
                    float(
                        test_rul_results["mae"]
                    ),

                "rmse":
                    float(
                        test_rul_results["rmse"]
                    ),

                "r2":
                    float(
                        test_rul_results["r2"]
                    ),

                "samples":
                    int(
                        X_test_lstm.shape[0]
                    )
            }
        ]
    )

    rul_path = (
        final_table_dir
        / "rul_prediction_results.csv"
    )

    rul_comparison.to_csv(
        rul_path,
        index=False
    )


    # ==========================================================
    # 9. DISPLAY RUL RESULTS
    # ==========================================================

    print("\n" + "-" * 70)
    print("FINAL RUL PREDICTION PERFORMANCE - TEST")
    print("-" * 70)

    print(
        f"Model : LSTM RUL Predictor"
    )

    print(
        f"MAE   : "
        f"{test_rul_results['mae']:.2f} cycles"
    )

    print(
        f"RMSE  : "
        f"{test_rul_results['rmse']:.2f} cycles"
    )

    print(
        f"R²    : "
        f"{test_rul_results['r2']:.4f}"
    )

    print(
        f"Samples : "
        f"{X_test_lstm.shape[0]}"
    )

    print(
        f"\n✓ Saved:"
        f"\n  {rul_path}"
    )


    # ==========================================================
    # 10. DATASET / PIPELINE SUMMARY
    # ==========================================================

    pipeline_summary = pd.DataFrame(
        [
            {
                "training_engines":
                    int(
                        train_df[
                            "engine_id"
                        ].nunique()
                    ),

                "validation_engines":
                    int(
                        validation_df[
                            "engine_id"
                        ].nunique()
                    ),

                "test_engines":
                    int(
                        test_df[
                            "engine_id"
                        ].nunique()
                    ),

                "training_windows":
                    int(
                        X_train.shape[0]
                    ),

                "validation_windows":
                    int(
                        X_validation.shape[0]
                    ),

                "test_windows":
                    int(
                        X_test.shape[0]
                    ),

                "window_length":
                    int(
                        X_train.shape[1]
                    ),

                "sensor_features":
                    int(
                        X_train.shape[2]
                    ),

                "wavelet_features":
                    int(
                        wavelet_train.shape[1]
                    )
            }
        ]
    )

    pipeline_summary_path = (
        final_table_dir
        / "pipeline_summary.csv"
    )

    pipeline_summary.to_csv(
        pipeline_summary_path,
        index=False
    )


    # ==========================================================
    # 11. FINAL RESEARCH SUMMARY
    # ==========================================================

    final_summary = pd.DataFrame(
        [
            {
                "component":
                    "Isolation Forest",

                "test_accuracy":
                    isolation_forest_test[
                        "accuracy"
                    ],

                "test_f1":
                    isolation_forest_test[
                        "f1_score"
                    ],

                "test_roc_auc":
                    isolation_forest_test[
                        "roc_auc"
                    ]
            },

            {
                "component":
                    "Dense Autoencoder",

                "test_accuracy":
                    autoencoder_test[
                        "accuracy"
                    ],

                "test_f1":
                    autoencoder_test[
                        "f1_score"
                    ],

                "test_roc_auc":
                    autoencoder_test[
                        "roc_auc"
                    ]
            },

            {
                "component":
                    "Proposed Fusion",

                "test_accuracy":
                    proposed_fusion_test[
                        "accuracy"
                    ],

                "test_f1":
                    proposed_fusion_test[
                        "f1_score"
                    ],

                "test_roc_auc":
                    proposed_fusion_test[
                        "roc_auc"
                    ]
            },

            {
                "component":
                    "LSTM RUL Predictor",

                "test_accuracy":
                    np.nan,

                "test_f1":
                    np.nan,

                "test_roc_auc":
                    np.nan
            }
        ]
    )

    final_summary_path = (
        final_table_dir
        / "final_research_summary.csv"
    )

    final_summary.to_csv(
        final_summary_path,
        index=False
    )


    # ==========================================================
    # 12. PRESENTATION-READY SUMMARY
    # ==========================================================

    print("\n" + "=" * 70)
    print("PRESENTATION-READY RESULTS")
    print("=" * 70)

    print("\nFAULT DETECTION")

    print(
        f"  Isolation Forest"
        f"  : Accuracy "
        f"{isolation_forest_test['accuracy'] * 100:.2f}%"
        f" | F1 "
        f"{isolation_forest_test['f1_score'] * 100:.2f}%"
        f" | AUC "
        f"{isolation_forest_test['roc_auc'] * 100:.2f}%"
    )

    print(
        f"  Autoencoder"
        f"        : Accuracy "
        f"{autoencoder_test['accuracy'] * 100:.2f}%"
        f" | F1 "
        f"{autoencoder_test['f1_score'] * 100:.2f}%"
        f" | AUC "
        f"{autoencoder_test['roc_auc'] * 100:.2f}%"
    )

    print(
        f"  Proposed Fusion"
        f"    : Accuracy "
        f"{proposed_fusion_test['accuracy'] * 100:.2f}%"
        f" | F1 "
        f"{proposed_fusion_test['f1_score'] * 100:.2f}%"
        f" | AUC "
        f"{proposed_fusion_test['roc_auc'] * 100:.2f}%"
    )

    print("\nRUL PREDICTION")

    print(
        f"  LSTM MAE  : "
        f"{test_rul_results['mae']:.2f} cycles"
    )

    print(
        f"  LSTM RMSE : "
        f"{test_rul_results['rmse']:.2f} cycles"
    )

    print(
        f"  LSTM R²   : "
        f"{test_rul_results['r2']:.4f}"
    )

    print("\nDATASET / PIPELINE")

    print(
        f"  Engines  : "
        f"{train_df['engine_id'].nunique()} train / "
        f"{validation_df['engine_id'].nunique()} validation / "
        f"{test_df['engine_id'].nunique()} test"
    )

    print(
        f"  Windows  : "
        f"{X_train.shape[0]} train / "
        f"{X_validation.shape[0]} validation / "
        f"{X_test.shape[0]} test"
    )

    print(
        f"  Window length : "
        f"{X_train.shape[1]} timesteps"
    )

    print(
        f"  Sensor features : "
        f"{X_train.shape[2]}"
    )

    print(
        f"  Wavelet features : "
        f"{wavelet_train.shape[1]}"
    )


    # ==========================================================
    # 13. FINAL STATUS
    # ==========================================================

    print("\n" + "=" * 70)
    print("STEP 15 COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        "\n✓ Fault detection comparison saved."
    )

    print(
        "✓ RUL prediction results saved."
    )

    print(
        "✓ Pipeline summary saved."
    )

    print(
        "✓ Presentation-ready results generated."
    )

    print(
        f"\nFinal output directory:"
        f"\n  {final_table_dir}"
    )
 
    

if __name__ == "__main__":
    main()
    