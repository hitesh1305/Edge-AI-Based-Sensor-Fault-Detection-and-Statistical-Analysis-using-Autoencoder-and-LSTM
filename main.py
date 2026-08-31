"""
Edge AI Fault Detection + RUL Prediction
NASA C-MAPSS

Clean pipeline runner.

Pipeline:
1. Data loading
2. EDA
3. Preprocessing + RUL
4. Engine-wise split
5. Normalization
6. Noise filtering
7. Z-score detection
8. Kalman filtering
9. Sliding windows
10. Wavelet features
11. Isolation Forest
12. Dense Autoencoder
13. Fault-score fusion
14. LSTM RUL prediction
15. Final results
"""

import os
import io
import contextlib

# Keep TensorFlow messages quiet
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score


# ============================================================
# PROJECT IMPORTS
# ============================================================

from configs.config import (
    TRAIN_FILE,
    TEST_FILE,
    RUL_FILE,
    WINDOW_SIZE,
)

from src.data.loader import CMAPSSLoader
from src.data.preprocessing import Preprocessor
from src.data.normalizer import DataNormalizer
from src.data.splitter import EngineWiseSplitter

from src.visualization.eda_visualizer import EDAVisualizer

from src.signal_processing.noise_filter import NoiseFilter
from src.signal_processing.kalman_filter import KalmanFilterEstimator

from src.statistics.zscore_detector import ZScoreDetector

from src.features.windowing import WindowGenerator
from src.features.wavelet_features import WaveletFeatureExtractor

from src.models.isolation_forest import IsolationForestDetector
from src.models.autoencoder import AutoencoderDetector
from src.models.fault_fusion import FaultScoreFusion
from src.models.lstm_predictor import LSTMPredictor


# ============================================================
# OUTPUT HELPERS
# ============================================================

def quiet_call(func, *args, **kwargs):
    """
    Run an existing project method without displaying
    its internal terminal output.

    Exceptions are NOT hidden.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


def step(number, text):
    print(f"[{number:02}] {text}")


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("Edge AI Fault Detection")
    print("NASA C-MAPSS | fault detection + RUL prediction")
    print()

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    step(1, "Loading data...")

    loader = CMAPSSLoader(
        TRAIN_FILE,
        TEST_FILE,
        RUL_FILE
    )

    train = quiet_call(
        loader.load_train
    )

    print(
        f"     {len(train):,} rows loaded"
    )

    # ========================================================
    # 2. EDA
    # ========================================================

    step(2, "Running EDA...")

    visualizer = EDAVisualizer(train)

    quiet_call(
        visualizer.engine_life_distribution
    )

    quiet_call(
        visualizer.sensor_variance
    )

    quiet_call(
        visualizer.correlation_heatmap
    )

    print("     plots saved")

    # ========================================================
    # 3. PREPROCESSING + RUL
    # ========================================================

    step(3, "Preprocessing + RUL...")

    preprocessor = Preprocessor()

    train = quiet_call(
        preprocessor.remove_constant_features,
        train
    )

    train = quiet_call(
        preprocessor.add_rul,
        train
    )

    print(
        f"     {train.shape[0]:,} rows | "
        f"RUL 0-{int(train['RUL'].max())}"
    )

    # ========================================================
    # 4. ENGINE-WISE SPLIT
    # ========================================================

    step(4, "Engine-wise split...")

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
    ) = quiet_call(
        splitter.split,
        train
    )

    print(
        f"     engines: "
        f"{train_df.engine_id.nunique()} / "
        f"{validation_df.engine_id.nunique()} / "
        f"{test_df.engine_id.nunique()}"
    )

    # ========================================================
    # 5. NORMALIZATION
    # ========================================================

    step(5, "Normalizing features...")

    normalizer = DataNormalizer()

    # Fit ONLY on training data
    quiet_call(
        normalizer.fit,
        train_df
    )

    train_df = quiet_call(
        normalizer.transform,
        train_df
    )

    validation_df = quiet_call(
        normalizer.transform,
        validation_df
    )

    test_df = quiet_call(
        normalizer.transform,
        test_df
    )

    quiet_call(
        normalizer.save,
        "output/models/scaler.pkl"
    )

    print(
        "     scaler fitted on training data"
    )

    # ========================================================
    # 6. NOISE FILTERING
    # ========================================================

    step(6, "Filtering sensor noise...")

    noise_filter = NoiseFilter()

    train_df = quiet_call(
        noise_filter.apply,
        train_df
    )

    validation_df = quiet_call(
        noise_filter.apply,
        validation_df
    )

    test_df = quiet_call(
        noise_filter.apply,
        test_df
    )

    print(
        "     train / validation / test done"
    )

    # ========================================================
    # 7. Z-SCORE ANALYSIS
    # ========================================================

    step(7, "Z-score analysis...")

    z_train = ZScoreDetector(
        threshold=3.0
    )

    z_validation = ZScoreDetector(
        threshold=3.0
    )

    z_test = ZScoreDetector(
        threshold=3.0
    )

    train_df = quiet_call(
        z_train.detect,
        train_df
    )

    validation_df = quiet_call(
        z_validation.detect,
        validation_df
    )

    test_df = quiet_call(
        z_test.detect,
        test_df
    )

    print(
        "     threshold = 3.0"
    )

    # ========================================================
    # 8. KALMAN FILTER
    # ========================================================

    step(8, "Kalman filtering...")

    kalman_train = KalmanFilterEstimator()
    kalman_validation = KalmanFilterEstimator()
    kalman_test = KalmanFilterEstimator()

    train_df = quiet_call(
        kalman_train.apply,
        train_df,
        split_name="train"
    )

    validation_df = quiet_call(
        kalman_validation.apply,
        validation_df,
        split_name="validation"
    )

    test_df = quiet_call(
        kalman_test.apply,
        test_df,
        split_name="test"
    )

    print(
        "     state estimates updated"
    )

    # ========================================================
    # 9. SLIDING WINDOWS
    # ========================================================

    step(9, "Creating sliding windows...")

    window_generator = WindowGenerator(
        window_size=WINDOW_SIZE,
        stride=1
    )

    (
        X_train,
        y_train,
        metadata_train
    ) = quiet_call(
        window_generator.generate,
        train_df,
        split_name="train"
    )

    (
        X_validation,
        y_validation,
        metadata_validation
    ) = quiet_call(
        window_generator.generate,
        validation_df,
        split_name="validation"
    )

    (
        X_test,
        y_test,
        metadata_test
    ) = quiet_call(
        window_generator.generate,
        test_df,
        split_name="test"
    )

    print(
        f"     windows: "
        f"{len(X_train):,} / "
        f"{len(X_validation):,} / "
        f"{len(X_test):,}"
    )

    print(
        f"     shape: "
        f"{X_train.shape[1]} timesteps x "
        f"{X_train.shape[2]} features"
    )

    # ========================================================
    # 10. WAVELET FEATURES
    # ========================================================

    step(10, "Extracting wavelet features...")

    sensor_columns = [
        column
        for column in train_df.columns
        if column.startswith("sensor_")
    ]

    # --------------------------
    # Training
    # --------------------------

    wavelet_train_extractor = (
        WaveletFeatureExtractor()
    )

    quiet_call(
        wavelet_train_extractor.set_input,
        X_train
    )

    wavelet_train = quiet_call(
        wavelet_train_extractor.extract,
        X_train,
        sensor_columns,
        split_name="train"
    )

    # --------------------------
    # Validation
    # --------------------------

    wavelet_validation_extractor = (
        WaveletFeatureExtractor()
    )

    quiet_call(
        wavelet_validation_extractor.set_input,
        X_validation
    )

    wavelet_validation = quiet_call(
        wavelet_validation_extractor.extract,
        X_validation,
        sensor_columns,
        split_name="validation"
    )

    # --------------------------
    # Test
    # --------------------------

    wavelet_test_extractor = (
        WaveletFeatureExtractor()
    )

    quiet_call(
        wavelet_test_extractor.set_input,
        X_test
    )

    wavelet_test = quiet_call(
        wavelet_test_extractor.extract,
        X_test,
        sensor_columns,
        split_name="test"
    )

    print(
        f"     features: "
        f"{wavelet_train.shape[1] - 1}"
    )

    # ========================================================
    # 11. ISOLATION FOREST
    # ========================================================

    step(11, "Isolation Forest...")

    isolation_forest = IsolationForestDetector(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        random_state=42,
        n_jobs=-1
    )

    # --------------------------
    # Train
    # --------------------------

    quiet_call(
        isolation_forest.fit,
        wavelet_train
    )

    # --------------------------
    # Validation scores
    # --------------------------

    validation_if_scores = quiet_call(
        isolation_forest.anomaly_scores,
        wavelet_validation
    )

    validation_if_labels = quiet_call(
        isolation_forest.create_rul_labels,
        metadata_validation,
        rul_threshold=30
    )

    # --------------------------
    # Validation threshold
    # --------------------------

    quiet_call(
        isolation_forest.select_threshold,
        validation_if_scores,
        validation_if_labels
    )

    validation_if_predictions = (
        np.asarray(validation_if_scores)
        >= isolation_forest.threshold
    ).astype(int)

    isolation_validation_results = quiet_call(
        isolation_forest.evaluate,
        validation_if_labels,
        validation_if_predictions,
        validation_if_scores,
        "validation"
    )

    # --------------------------
    # Test
    # --------------------------

    test_if_predictions, test_if_scores = (
        quiet_call(
            isolation_forest.predict,
            wavelet_test
        )
    )

    test_if_labels = quiet_call(
        isolation_forest.create_rul_labels,
        metadata_test,
        rul_threshold=30
    )

    isolation_test_results = quiet_call(
        isolation_forest.evaluate,
        test_if_labels,
        test_if_predictions,
        test_if_scores,
        "test"
    )

    # --------------------------
    # Save results
    # --------------------------

    quiet_call(
        isolation_forest.save_results,
        isolation_validation_results
    )

    quiet_call(
        isolation_forest.save_results,
        isolation_test_results
    )

    quiet_call(
        isolation_forest.plot_confusion_matrix,
        test_if_labels,
        test_if_predictions,
        "test"
    )

    quiet_call(
        isolation_forest.plot_roc_curve,
        test_if_labels,
        test_if_scores,
        "test"
    )

    quiet_call(
        isolation_forest.plot_score_distribution,
        test_if_scores,
        test_if_labels,
        "test",
        isolation_forest.threshold
    )

    quiet_call(
        isolation_forest.measure_inference_time,
        wavelet_test
    )

    print(
        f"     test F1 "
        f"{isolation_test_results['f1_score']:.4f} | "
        f"AUC "
        f"{isolation_test_results['roc_auc']:.4f}"
    )

    # ========================================================
    # 12. DENSE AUTOENCODER
    # ========================================================

    step(12, "Dense Autoencoder...")

    # --------------------------------------------------------
    # Ground-truth labels
    # --------------------------------------------------------

    train_labels = (
        metadata_train["target_rul"] <= 30
    ).astype(int).to_numpy()

    validation_labels = (
        metadata_validation["target_rul"] <= 30
    ).astype(int).to_numpy()

    test_labels = (
        metadata_test["target_rul"] <= 30
    ).astype(int).to_numpy()

    # --------------------------------------------------------
    # Convert wavelet data to NumPy
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Healthy training data
    # --------------------------------------------------------

    healthy_train = (
        X_wavelet_train[
            train_labels == 0
        ]
    )

    healthy_validation = (
        X_wavelet_validation[
            validation_labels == 0
        ]
    )

    # --------------------------------------------------------
    # Create Autoencoder
    # --------------------------------------------------------

    autoencoder = AutoencoderDetector(
        input_dim=X_wavelet_train.shape[1],
        encoding_dim=64,
        learning_rate=0.0005,
        batch_size=128,
        epochs=50
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    history, _ = quiet_call(
        autoencoder.fit,
        healthy_train,
        healthy_validation
    )

    quiet_call(
        autoencoder.plot_training_history,
        history
    )

    # --------------------------------------------------------
    # Validation reconstruction errors
    # --------------------------------------------------------

    validation_ae_errors = quiet_call(
        autoencoder.reconstruction_errors,
        X_wavelet_validation
    )

    # --------------------------------------------------------
    # Select threshold using validation
    # --------------------------------------------------------

    quiet_call(
        autoencoder.select_threshold,
        validation_ae_errors,
        validation_labels
    )

    validation_ae_predictions = (
        np.asarray(validation_ae_errors)
        >= autoencoder.threshold
    ).astype(int)

    autoencoder_validation_results = (
        quiet_call(
            autoencoder.evaluate,
            validation_labels,
            validation_ae_predictions,
            validation_ae_errors,
            "validation"
        )
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_ae_predictions, test_ae_errors = (
        quiet_call(
            autoencoder.predict,
            X_wavelet_test
        )
    )

    autoencoder_test_results = quiet_call(
        autoencoder.evaluate,
        test_labels,
        test_ae_predictions,
        test_ae_errors,
        "test"
    )

    # --------------------------------------------------------
    # Save Autoencoder results
    # --------------------------------------------------------

    quiet_call(
        autoencoder.save_results,
        autoencoder_validation_results
    )

    quiet_call(
        autoencoder.save_results,
        autoencoder_test_results
    )

    quiet_call(
        autoencoder.plot_error_distribution,
        test_ae_errors,
        test_labels,
        autoencoder.threshold,
        "test"
    )

    quiet_call(
        autoencoder.plot_confusion_matrix,
        test_labels,
        test_ae_predictions,
        "test"
    )

    quiet_call(
        autoencoder.plot_roc_curve,
        test_labels,
        test_ae_errors,
        "test"
    )

    quiet_call(
        autoencoder.measure_inference_time,
        X_wavelet_test
    )

    quiet_call(
        autoencoder.save_model
    )

    print(
        f"     test F1 "
        f"{autoencoder_test_results['f1_score']:.4f} | "
        f"AUC "
        f"{autoencoder_test_results['roc_auc']:.4f}"
    )

    # ========================================================
    # 13. PROPOSED FAULT-SCORE FUSION
    # ========================================================

    step(13, "Fault-score fusion...")

    """
    Proposed fusion:

        Isolation Forest
                +
        Dense Autoencoder
                |
          score alignment
                |
        validation normalization
                |
          weighted fusion
                |
        validation threshold
                |
          final test result

    Validation data is used for normalization and
    threshold selection.

    Test data is used only for final evaluation.
    """

    # --------------------------------------------------------
    # Create fusion model
    # --------------------------------------------------------

    fault_fusion = FaultScoreFusion(
        isolation_weight=0.5,
        autoencoder_weight=0.5
    )

    # --------------------------------------------------------
    # Prepare arrays
    # --------------------------------------------------------

    validation_if_scores = np.asarray(
        validation_if_scores,
        dtype=np.float64
    ).reshape(-1)

    validation_ae_errors = np.asarray(
        validation_ae_errors,
        dtype=np.float64
    ).reshape(-1)

    validation_labels_array = np.asarray(
        validation_labels,
        dtype=int
    ).reshape(-1)

    test_if_scores = np.asarray(
        test_if_scores,
        dtype=np.float64
    ).reshape(-1)

    test_ae_errors = np.asarray(
        test_ae_errors,
        dtype=np.float64
    ).reshape(-1)

    test_labels_array = np.asarray(
        test_labels,
        dtype=int
    ).reshape(-1)

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if not (
        len(validation_if_scores)
        == len(validation_ae_errors)
        == len(validation_labels_array)
    ):
        raise ValueError(
            "Validation fusion arrays have "
            "different lengths."
        )

    if not (
        len(test_if_scores)
        == len(test_ae_errors)
        == len(test_labels_array)
    ):
        raise ValueError(
            "Test fusion arrays have "
            "different lengths."
        )

    # --------------------------------------------------------
    # Align Isolation Forest score direction
    # --------------------------------------------------------
    #
    # Higher score must mean:
    # higher probability of anomaly.
    #
    # The direction is determined ONLY from
    # validation ROC-AUC.

    raw_auc = roc_auc_score(
        validation_labels_array,
        validation_if_scores
    )

    reversed_auc = roc_auc_score(
        validation_labels_array,
        -validation_if_scores
    )

    if reversed_auc > raw_auc:

        validation_if_aligned = (
            -validation_if_scores
        )

        test_if_aligned = (
            -test_if_scores
        )

    else:

        validation_if_aligned = (
            validation_if_scores
        )

        test_if_aligned = (
            test_if_scores
        )

    # --------------------------------------------------------
    # Normalize scores
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # Fit ONLY on validation data.
    #
    # Do NOT fit the scaler on test data.

    fault_fusion.fit_scaler(
        validation_if_aligned,
        validation_ae_errors
    )

    (
        validation_if_normalized,
        validation_ae_normalized
    ) = fault_fusion.transform_scores(
        validation_if_aligned,
        validation_ae_errors
    )

    # --------------------------------------------------------
    # Validation fused score
    # --------------------------------------------------------

    validation_fused_scores = (
        fault_fusion.fuse_scores(
            validation_if_normalized,
            validation_ae_normalized
        )
    )

    # --------------------------------------------------------
    # Select fusion threshold
    # --------------------------------------------------------

    fault_fusion.select_threshold(
        validation_fused_scores,
        validation_labels_array
    )

    validation_fusion_predictions = (
        fault_fusion.predict(
            validation_fused_scores
        )
    )

    # --------------------------------------------------------
    # Validation evaluation
    # --------------------------------------------------------

    fusion_validation_results = (
        fault_fusion.evaluate(
            validation_labels_array,
            validation_fusion_predictions,
            validation_fused_scores,
            "validation"
        )
    )

    # --------------------------------------------------------
    # TEST NORMALIZATION
    # --------------------------------------------------------
    #
    # Reuse validation-fitted scaler.

    (
        test_if_normalized,
        test_ae_normalized
    ) = fault_fusion.transform_scores(
        test_if_aligned,
        test_ae_errors
    )

    # --------------------------------------------------------
    # Test fused score
    # --------------------------------------------------------

    test_fused_scores = (
        fault_fusion.fuse_scores(
            test_if_normalized,
            test_ae_normalized
        )
    )

    # --------------------------------------------------------
    # Test prediction
    # --------------------------------------------------------

    test_fusion_predictions = (
        fault_fusion.predict(
            test_fused_scores
        )
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    fusion_test_results = (
        fault_fusion.evaluate(
            test_labels_array,
            test_fusion_predictions,
            test_fused_scores,
            "test"
        )
    )

    # --------------------------------------------------------
    # Save validation fusion scores
    # --------------------------------------------------------

    fault_fusion.save_scores(
        window_ids=(
            metadata_validation[
                "window_id"
            ].to_numpy()
        ),
        labels=validation_labels_array,
        isolation_scores=validation_if_aligned,
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

    # --------------------------------------------------------
    # Save test fusion scores
    # --------------------------------------------------------

    fault_fusion.save_scores(
        window_ids=(
            metadata_test[
                "window_id"
            ].to_numpy()
        ),
        labels=test_labels_array,
        isolation_scores=test_if_aligned,
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

    # --------------------------------------------------------
    # Save final fusion results
    # --------------------------------------------------------

    fault_fusion.save_results(
        fusion_validation_results,
        "output/tables/fusion/"
        "validation_results.csv"
    )

    fault_fusion.save_results(
        fusion_test_results,
        "output/tables/fusion/"
        "results.csv"
    )

    # --------------------------------------------------------
    # Fusion figures
    # --------------------------------------------------------

    fault_fusion.plot_confusion_matrix(
        test_labels_array,
        test_fusion_predictions,
        "output/figures/"
        "fusion_confusion_test.png"
    )

    fault_fusion.plot_roc_curve(
        test_labels_array,
        test_fused_scores,
        "output/figures/"
        "fusion_roc_test.png"
    )

    fault_fusion.plot_score_distribution(
        test_fused_scores,
        test_labels_array,
        fault_fusion.threshold,
        "output/figures/"
        "fusion_score_distribution_test.png"
    )

    # --------------------------------------------------------
    # Fusion inference
    # --------------------------------------------------------

    quiet_call(
        fault_fusion.measure_inference_time,
        test_if_aligned,
        test_ae_errors
    )

    print(
        f"     test F1 "
        f"{fusion_test_results['f1_score']:.4f} | "
        f"AUC "
        f"{fusion_test_results['roc_auc']:.4f}"
    )

    # ========================================================
    # 14. LSTM RUL PREDICTION
    # ========================================================

    step(14, "LSTM RUL prediction...")

    # --------------------------------------------------------
    # LSTM input
    # --------------------------------------------------------

    X_train_lstm = (
        X_train.astype(np.float32)
    )

    X_validation_lstm = (
        X_validation.astype(np.float32)
    )

    X_test_lstm = (
        X_test.astype(np.float32)
    )

    # --------------------------------------------------------
    # RUL targets
    # --------------------------------------------------------

    y_train_lstm = (
        metadata_train[
            "target_rul"
        ].to_numpy(
            dtype=np.float32
        )
    )

    y_validation_lstm = (
        metadata_validation[
            "target_rul"
        ].to_numpy(
            dtype=np.float32
        )
    )

    y_test_lstm = (
        metadata_test[
            "target_rul"
        ].to_numpy(
            dtype=np.float32
        )
    )

    # --------------------------------------------------------
    # Create predictor
    # --------------------------------------------------------

    lstm_predictor = LSTMPredictor(
        timesteps=X_train_lstm.shape[1],
        features=X_train_lstm.shape[2],
        lstm_units=64,
        learning_rate=0.001,
        batch_size=128,
        epochs=50,
        dropout=0.2,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    lstm_history, _ = quiet_call(
        lstm_predictor.fit,
        X_train_lstm,
        y_train_lstm,
        X_validation_lstm,
        y_validation_lstm
    )

    quiet_call(
        lstm_predictor.plot_training_history,
        lstm_history
    )

    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    validation_rul_pred = quiet_call(
        lstm_predictor.predict,
        X_validation_lstm
    )

    validation_rul_results = quiet_call(
        lstm_predictor.evaluate,
        y_validation_lstm,
        validation_rul_pred,
        "validation"
    )

    # --------------------------------------------------------
    # Test prediction
    # --------------------------------------------------------

    test_rul_pred = quiet_call(
        lstm_predictor.predict,
        X_test_lstm
    )

    test_rul_results = quiet_call(
        lstm_predictor.evaluate,
        y_test_lstm,
        test_rul_pred,
        "test"
    )

    # --------------------------------------------------------
    # Required figure
    # --------------------------------------------------------

    quiet_call(
        lstm_predictor.plot_actual_vs_predicted,
        y_test_lstm,
        test_rul_pred,
        "test"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    quiet_call(
        lstm_predictor.save_results,
        validation_rul_results
    )

    quiet_call(
        lstm_predictor.save_results,
        test_rul_results
    )

    quiet_call(
        lstm_predictor.save_model
    )

    print(
        f"     MAE "
        f"{test_rul_results['mae']:.2f} | "
        f"RMSE "
        f"{test_rul_results['rmse']:.2f} | "
        f"R2 "
        f"{test_rul_results['r2']:.3f}"
    )

    # ========================================================
    # 15. FINAL RESULTS
    # ========================================================

    step(15, "Final results")

    print()
    print("Fault detection")

    print(
        f"  Isolation Forest : "
        f"F1 "
        f"{isolation_test_results['f1_score']:.2%} | "
        f"AUC "
        f"{isolation_test_results['roc_auc']:.3f}"
    )

    print(
        f"  Autoencoder      : "
        f"F1 "
        f"{autoencoder_test_results['f1_score']:.2%} | "
        f"AUC "
        f"{autoencoder_test_results['roc_auc']:.3f}"
    )

    print(
        f"  Fusion           : "
        f"F1 "
        f"{fusion_test_results['f1_score']:.2%} | "
        f"AUC "
        f"{fusion_test_results['roc_auc']:.3f}"
    )

    print()
    print("RUL prediction")

    print(
        f"  LSTM             : "
        f"MAE "
        f"{test_rul_results['mae']:.2f} | "
        f"RMSE "
        f"{test_rul_results['rmse']:.2f} | "
        f"R2 "
        f"{test_rul_results['r2']:.3f}"
    )

    print()
    print(
        f"Data: "
        f"{len(X_train):,} train / "
        f"{len(X_validation):,} val / "
        f"{len(X_test):,} test windows"
    )

    print()
    print("Done.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()