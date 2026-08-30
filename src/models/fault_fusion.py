"""
Fault Score Fusion
==================

Research-grade fusion of:
    1. Isolation Forest anomaly score
    2. Dense Autoencoder reconstruction error

Features
--------
- Robust validation-based score calibration
- Validation-only fusion-weight optimization
- Validation-only threshold selection
- Final test evaluation without test-set tuning
- Accuracy, Precision, Recall, F1 and ROC-AUC
- Confusion matrix, ROC and score-distribution figures
- Inference-time measurement
- Raw and normalized score saving

Research Protocol
-----------------
TRAIN:
    Models are trained using training data.

VALIDATION:
    Score calibration parameters are estimated.
    Fusion weights are optimized.
    Fusion threshold is selected.

TEST:
    Validation calibration, weight and threshold are frozen.
    Test data is used only for final evaluation.

Important
---------
No test labels are used for:
    - score calibration
    - fusion-weight selection
    - threshold selection
"""

from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)


class FaultScoreFusion:

    def __init__(
        self,
        isolation_weight=0.5,
        autoencoder_weight=0.5,
        weight_step=0.05,
        calibration_low=1.0,
        calibration_high=99.0,
    ):
        """
        Parameters
        ----------
        isolation_weight : float
            Initial Isolation Forest weight.

        autoencoder_weight : float
            Initial Autoencoder weight.

        weight_step : float
            Step used during validation-based weight search.

        calibration_low : float
            Lower percentile used for robust score calibration.

        calibration_high : float
            Upper percentile used for robust score calibration.

        Notes
        -----
        The initial weights are used only as defaults.
        During the proposed experiment, optimize_weights()
        searches for the best weight combination using
        validation data only.
        """

        if isolation_weight < 0:
            raise ValueError(
                "Isolation Forest weight cannot be negative."
            )

        if autoencoder_weight < 0:
            raise ValueError(
                "Autoencoder weight cannot be negative."
            )

        total = (
            isolation_weight
            + autoencoder_weight
        )

        if total <= 0:
            raise ValueError(
                "Fusion weights must have a positive sum."
            )

        self.isolation_weight = (
            isolation_weight / total
        )

        self.autoencoder_weight = (
            autoencoder_weight / total
        )

        self.weight_step = float(
            weight_step
        )

        if not (0 < self.weight_step <= 1):
            raise ValueError(
                "weight_step must be in (0, 1]."
            )

        if not (
            0 <= calibration_low < calibration_high <= 100
        ):
            raise ValueError(
                "Invalid calibration percentiles."
            )

        self.calibration_low = (
            float(calibration_low)
        )

        self.calibration_high = (
            float(calibration_high)
        )

        # Calibration parameters learned from validation only.
        self.isolation_low = None
        self.isolation_high = None

        self.autoencoder_low = None
        self.autoencoder_high = None

        self.threshold = None

        # Store validation optimization information.
        self.best_validation_f1 = None
        self.best_validation_precision = None
        self.best_validation_recall = None

    # ======================================================
    # ROBUST SCORE CALIBRATION
    # ======================================================

    @staticmethod
    def _robust_parameters(
        scores,
        low_percentile,
        high_percentile,
    ):
        """
        Calculate robust lower/upper score bounds.

        Parameters are estimated only from the supplied
        validation scores.

        Extreme observations outside these limits are clipped.
        """

        scores = np.asarray(
            scores,
            dtype=np.float64,
        ).reshape(-1)

        if len(scores) == 0:
            raise ValueError(
                "Cannot calibrate empty score array."
            )

        low = np.percentile(
            scores,
            low_percentile,
        )

        high = np.percentile(
            scores,
            high_percentile,
        )

        # Prevent division by zero for constant scores.
        if np.isclose(
            high,
            low,
        ):
            high = low + 1e-12

        return (
            float(low),
            float(high),
        )

    @staticmethod
    def _apply_calibration(
        scores,
        low,
        high,
    ):
        """
        Map anomaly scores to [0, 1] using fixed
        validation-derived calibration parameters.
        """

        scores = np.asarray(
            scores,
            dtype=np.float64,
        )

        normalized = (
            scores - low
        ) / (
            high - low
        )

        normalized = np.clip(
            normalized,
            0.0,
            1.0,
        )

        return normalized

    def fit_scaler(
        self,
        isolation_scores,
        autoencoder_errors,
    ):
        """
        Fit robust score calibration using VALIDATION data only.

        Unlike MinMaxScaler, percentile calibration is less
        sensitive to extreme validation observations.

        No labels are required.
        """

        isolation_scores = np.asarray(
            isolation_scores,
            dtype=np.float64,
        ).reshape(-1)

        autoencoder_errors = np.asarray(
            autoencoder_errors,
            dtype=np.float64,
        ).reshape(-1)

        if len(isolation_scores) != len(
            autoencoder_errors
        ):
            raise ValueError(
                "Isolation Forest scores and "
                "Autoencoder errors must have "
                "the same number of samples."
            )

        (
            self.isolation_low,
            self.isolation_high,
        ) = self._robust_parameters(
            isolation_scores,
            self.calibration_low,
            self.calibration_high,
        )

        (
            self.autoencoder_low,
            self.autoencoder_high,
        ) = self._robust_parameters(
            autoencoder_errors,
            self.calibration_low,
            self.calibration_high,
        )

        print("\n" + "=" * 70)
        print("FUSION SCORE CALIBRATION")
        print("=" * 70)

        print(
            f"Isolation Forest calibration : "
            f"{self.isolation_low:.8f} "
            f"to "
            f"{self.isolation_high:.8f}"
        )

        print(
            f"Autoencoder calibration      : "
            f"{self.autoencoder_low:.8f} "
            f"to "
            f"{self.autoencoder_high:.8f}"
        )

        print(
            "\n✓ Calibration parameters learned "
            "from validation data only."
        )

        return self

    def transform_scores(
        self,
        isolation_scores,
        autoencoder_errors,
    ):
        """
        Apply validation-derived score calibration.

        No fitting occurs here.

        This method can therefore safely be used on:
            - validation data
            - test data
            - deployment/inference data
        """

        if (
            self.isolation_low is None
            or self.isolation_high is None
            or self.autoencoder_low is None
            or self.autoencoder_high is None
        ):
            raise RuntimeError(
                "Calibration parameters have not been fitted. "
                "Call fit_scaler() first."
            )

        isolation_scores = np.asarray(
            isolation_scores,
            dtype=np.float64,
        ).reshape(-1)

        autoencoder_errors = np.asarray(
            autoencoder_errors,
            dtype=np.float64,
        ).reshape(-1)

        if len(isolation_scores) != len(
            autoencoder_errors
        ):
            raise ValueError(
                "Isolation Forest scores and "
                "Autoencoder errors must have "
                "the same number of samples."
            )

        isolation_normalized = (
            self._apply_calibration(
                isolation_scores,
                self.isolation_low,
                self.isolation_high,
            )
        )

        autoencoder_normalized = (
            self._apply_calibration(
                autoencoder_errors,
                self.autoencoder_low,
                self.autoencoder_high,
            )
        )

        return (
            isolation_normalized,
            autoencoder_normalized,
        )

    # ======================================================
    # FUSION SCORE
    # ======================================================

    def fuse_scores(
        self,
        isolation_scores,
        autoencoder_errors,
    ):
        """
        Generate weighted anomaly score.

        Higher score = more anomalous.
        """

        isolation_scores = np.asarray(
            isolation_scores,
            dtype=np.float64,
        ).reshape(-1)

        autoencoder_errors = np.asarray(
            autoencoder_errors,
            dtype=np.float64,
        ).reshape(-1)

        if len(isolation_scores) != len(
            autoencoder_errors
        ):
            raise ValueError(
                "Fusion inputs must have equal length."
            )

        fused = (
            self.isolation_weight
            * isolation_scores
            +
            self.autoencoder_weight
            * autoencoder_errors
        )

        return fused

    # ======================================================
    # VALIDATION WEIGHT OPTIMIZATION
    # ======================================================

    def optimize_weights(
        self,
        isolation_scores,
        autoencoder_scores,
        validation_labels,
    ):
        """
        Optimize Isolation Forest / Autoencoder weights
        using VALIDATION data only.

        For every candidate weight:

            fused =
                w_IF * IF_score
                +
                w_AE * AE_score

        The threshold for that candidate is selected using
        validation F1.

        The best validation-F1 configuration is retained.

        IMPORTANT
        ---------
        Test data is never used by this method.
        """

        isolation_scores = np.asarray(
            isolation_scores,
            dtype=np.float64,
        ).reshape(-1)

        autoencoder_scores = np.asarray(
            autoencoder_scores,
            dtype=np.float64,
        ).reshape(-1)

        validation_labels = np.asarray(
            validation_labels,
            dtype=int,
        ).reshape(-1)

        if not (
            len(isolation_scores)
            == len(autoencoder_scores)
            == len(validation_labels)
        ):
            raise ValueError(
                "All validation arrays must have "
                "the same length."
            )

        print("\n" + "=" * 70)
        print("VALIDATION-BASED FUSION WEIGHT OPTIMIZATION")
        print("=" * 70)

        print(
            f"Weight Search Step : "
            f"{self.weight_step:.2f}"
        )

        best_weight = None
        best_f1 = -1.0
        best_precision = 0.0
        best_recall = 0.0
        best_threshold = None

        weight_results = []

        weights = np.arange(
            0.0,
            1.0 + self.weight_step / 2.0,
            self.weight_step,
        )

        for isolation_weight in weights:

            isolation_weight = float(
                min(
                    max(
                        isolation_weight,
                        0.0,
                    ),
                    1.0,
                )
            )

            autoencoder_weight = (
                1.0 - isolation_weight
            )

            fused_scores = (
                isolation_weight
                * isolation_scores
                +
                autoencoder_weight
                * autoencoder_scores
            )

            (
                threshold,
                current_f1,
                current_precision,
                current_recall,
            ) = self._find_best_threshold(
                fused_scores,
                validation_labels,
            )

            weight_results.append(
                {
                    "isolation_weight":
                        isolation_weight,

                    "autoencoder_weight":
                        autoencoder_weight,

                    "validation_f1":
                        current_f1,

                    "validation_precision":
                        current_precision,

                    "validation_recall":
                        current_recall,

                    "threshold":
                        threshold,
                }
            )

            # Primary optimization criterion:
            # validation F1.
            if current_f1 > best_f1:

                best_f1 = current_f1

                best_weight = (
                    isolation_weight
                )

                best_precision = (
                    current_precision
                )

                best_recall = (
                    current_recall
                )

                best_threshold = (
                    threshold
                )

        if best_weight is None:
            raise RuntimeError(
                "Fusion weight optimization failed."
            )

        self.isolation_weight = (
            float(best_weight)
        )

        self.autoencoder_weight = (
            float(1.0 - best_weight)
        )

        self.threshold = (
            float(best_threshold)
        )

        self.best_validation_f1 = (
            float(best_f1)
        )

        self.best_validation_precision = (
            float(best_precision)
        )

        self.best_validation_recall = (
            float(best_recall)
        )

        results_df = pd.DataFrame(
            weight_results
        )

        output_path = Path(
            "output/tables/fusion/"
            "weight_search.csv"
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        results_df.to_csv(
            output_path,
            index=False,
        )

        print(
            "\n✓ Optimal fusion weights selected."
        )

        print(
            f"\nSelected Isolation Forest Weight : "
            f"{self.isolation_weight:.2f}"
        )

        print(
            f"Selected Autoencoder Weight      : "
            f"{self.autoencoder_weight:.2f}"
        )

        print(
            f"Validation F1                    : "
            f"{self.best_validation_f1:.4f}"
        )

        print(
            f"Validation Precision             : "
            f"{self.best_validation_precision:.4f}"
        )

        print(
            f"Validation Recall                : "
            f"{self.best_validation_recall:.4f}"
        )

        print(
            f"Validation Threshold             : "
            f"{self.threshold:.8f}"
        )

        print(
            f"\n✓ Weight search saved:"
            f"\n  {output_path}"
        )

        return {
            "isolation_weight":
                self.isolation_weight,

            "autoencoder_weight":
                self.autoencoder_weight,

            "threshold":
                self.threshold,

            "validation_f1":
                self.best_validation_f1,

            "validation_precision":
                self.best_validation_precision,

            "validation_recall":
                self.best_validation_recall,
        }

    # ======================================================
    # INTERNAL THRESHOLD SEARCH
    # ======================================================

    @staticmethod
    def _find_best_threshold(
        scores,
        labels,
    ):
        """
        Find threshold maximizing F1.

        Used exclusively on validation data.
        """

        scores = np.asarray(
            scores,
            dtype=np.float64,
        ).reshape(-1)

        labels = np.asarray(
            labels,
            dtype=int,
        ).reshape(-1)

        unique_scores = np.unique(
            scores
        )

        if len(unique_scores) == 0:
            raise ValueError(
                "Cannot select threshold from empty scores."
            )

        if len(unique_scores) > 2000:

            thresholds = np.percentile(
                scores,
                np.linspace(
                    0.0,
                    100.0,
                    2001,
                ),
            )

            thresholds = np.unique(
                thresholds
            )

        else:

            thresholds = unique_scores

        best_threshold = None
        best_f1 = -1.0
        best_precision = 0.0
        best_recall = 0.0

        for threshold in thresholds:

            predictions = (
                scores >= threshold
            ).astype(int)

            current_f1 = f1_score(
                labels,
                predictions,
                zero_division=0,
            )

            if current_f1 > best_f1:

                best_f1 = (
                    current_f1
                )

                best_threshold = (
                    float(threshold)
                )

                best_precision = (
                    precision_score(
                        labels,
                        predictions,
                        zero_division=0,
                    )
                )

                best_recall = (
                    recall_score(
                        labels,
                        predictions,
                        zero_division=0,
                    )
                )

        return (
            best_threshold,
            best_f1,
            best_precision,
            best_recall,
        )

    # ======================================================
    # THRESHOLD SELECTION
    # ======================================================

    def select_threshold(
        self,
        fused_validation_scores,
        validation_labels,
    ):
        """
        Select threshold using validation F1.

        The test set is NOT used.
        """

        (
            best_threshold,
            best_f1,
            best_precision,
            best_recall,
        ) = self._find_best_threshold(
            fused_validation_scores,
            validation_labels,
        )

        self.threshold = (
            float(best_threshold)
        )

        self.best_validation_f1 = (
            float(best_f1)
        )

        self.best_validation_precision = (
            float(best_precision)
        )

        self.best_validation_recall = (
            float(best_recall)
        )

        print("\n" + "=" * 70)
        print("FUSION THRESHOLD SELECTION")
        print("=" * 70)

        print(
            f"Selected Threshold : "
            f"{self.threshold:.8f}"
        )

        print(
            f"Validation F1      : "
            f"{best_f1:.4f}"
        )

        print(
            f"Validation Precision: "
            f"{best_precision:.4f}"
        )

        print(
            f"Validation Recall   : "
            f"{best_recall:.4f}"
        )

        return self.threshold

    # ======================================================
    # PREDICTION
    # ======================================================

    def predict(
        self,
        fused_scores,
    ):
        """
        Convert fused anomaly scores into binary predictions.
        """

        if self.threshold is None:
            raise RuntimeError(
                "Threshold has not been selected."
            )

        scores = np.asarray(
            fused_scores,
            dtype=np.float64,
        )

        return (
            scores >= self.threshold
        ).astype(int)

    # ======================================================
    # EVALUATION
    # ======================================================

    def evaluate(
        self,
        labels,
        predictions,
        scores,
        split_name,
    ):
        """
        Evaluate detector performance.
        """

        labels = np.asarray(
            labels,
            dtype=int,
        )

        predictions = np.asarray(
            predictions,
            dtype=int,
        )

        scores = np.asarray(
            scores,
            dtype=np.float64,
        )

        accuracy = accuracy_score(
            labels,
            predictions,
        )

        precision = precision_score(
            labels,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            labels,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            labels,
            predictions,
            zero_division=0,
        )

        try:

            roc_auc = roc_auc_score(
                labels,
                scores,
            )

        except ValueError:

            roc_auc = np.nan

        cm = confusion_matrix(
            labels,
            predictions,
        )

        print("\n" + "-" * 70)
        print(
            f"PROPOSED FUSION - "
            f"{split_name.upper()}"
        )
        print("-" * 70)

        print(
            f"Accuracy  : {accuracy:.4f}"
        )

        print(
            f"Precision : {precision:.4f}"
        )

        print(
            f"Recall    : {recall:.4f}"
        )

        print(
            f"F1 Score  : {f1:.4f}"
        )

        print(
            f"ROC-AUC   : {roc_auc:.4f}"
        )

        print(
            "\nConfusion Matrix:"
        )

        print(cm)

        results = {
            "model":
                "Proposed Score Fusion",

            "split":
                split_name,

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1_score":
                f1,

            "roc_auc":
                roc_auc,

            "threshold":
                self.threshold,

            "samples":
                len(labels),

            "actual_anomalies":
                int(np.sum(labels == 1)),

            "predicted_anomalies":
                int(np.sum(predictions == 1)),

            "isolation_weight":
                self.isolation_weight,

            "autoencoder_weight":
                self.autoencoder_weight,

            "calibration_low_percentile":
                self.calibration_low,

            "calibration_high_percentile":
                self.calibration_high,
        }

        return results

    # ======================================================
    # SAVE SCORES
    # ======================================================

    @staticmethod
    def save_scores(
        window_ids,
        labels,
        isolation_scores,
        autoencoder_errors,
        isolation_normalized,
        autoencoder_normalized,
        fused_scores,
        predictions,
        output_path,
    ):
        """
        Save all fusion-related scores for reproducibility.
        """

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = pd.DataFrame(
            {
                "window_id":
                    window_ids,

                "actual_label":
                    labels,

                "isolation_forest_score":
                    isolation_scores,

                "autoencoder_error":
                    autoencoder_errors,

                "isolation_normalized":
                    isolation_normalized,

                "autoencoder_normalized":
                    autoencoder_normalized,

                "fused_score":
                    fused_scores,

                "fusion_prediction":
                    predictions,
            }
        )

        data.to_csv(
            output_path,
            index=False,
        )

        print(
            f"\n✓ Fusion scores saved:"
            f"\n  {output_path}"
        )

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    @staticmethod
    def save_results(
        results,
        output_path,
    ):
        """
        Save evaluation results.
        """

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        pd.DataFrame(
            [results]
        ).to_csv(
            output_path,
            index=False,
        )

        print(
            f"\n✓ Fusion results saved:"
            f"\n  {output_path}"
        )

    # ======================================================
    # CONFUSION MATRIX
    # ======================================================

    @staticmethod
    def plot_confusion_matrix(
        labels,
        predictions,
        output_path,
        title="Proposed Fusion - Confusion Matrix",
    ):
        """
        Save confusion matrix figure.
        """

        cm = confusion_matrix(
            labels,
            predictions,
        )

        plt.figure(
            figsize=(7, 6)
        )

        plt.imshow(
            cm,
            interpolation="nearest",
        )

        plt.title(
            title
        )

        plt.colorbar()

        tick_marks = np.arange(
            2
        )

        plt.xticks(
            tick_marks,
            [
                "Normal",
                "Fault",
            ],
        )

        plt.yticks(
            tick_marks,
            [
                "Normal",
                "Fault",
            ],
        )

        for i in range(
            cm.shape[0]
        ):

            for j in range(
                cm.shape[1]
            ):

                plt.text(
                    j,
                    i,
                    str(cm[i, j]),
                    horizontalalignment="center",
                    verticalalignment="center",
                )

        plt.xlabel(
            "Predicted Label"
        )

        plt.ylabel(
            "Actual Label"
        )

        plt.tight_layout()

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"✓ Fusion confusion matrix saved:"
            f"\n  {output_path}"
        )

    # ======================================================
    # ROC CURVE
    # ======================================================

    @staticmethod
    def plot_roc_curve(
        labels,
        scores,
        output_path,
        title="Proposed Fusion - ROC Curve",
    ):
        """
        Save ROC curve.
        """

        fpr, tpr, _ = roc_curve(
            labels,
            scores,
        )

        auc = roc_auc_score(
            labels,
            scores,
        )

        plt.figure(
            figsize=(8, 6)
        )

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=(
                f"Fusion ROC-AUC = "
                f"{auc:.4f}"
            ),
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            linewidth=1.5,
        )

        plt.xlabel(
            "False Positive Rate"
        )

        plt.ylabel(
            "True Positive Rate"
        )

        plt.title(
            title
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"✓ Fusion ROC curve saved:"
            f"\n  {output_path}"
        )

    # ======================================================
    # SCORE DISTRIBUTION
    # ======================================================

    @staticmethod
    def plot_score_distribution(
        scores,
        labels,
        threshold,
        output_path,
        title="Proposed Fusion Score Distribution",
    ):
        """
        Save normal/fault fusion-score distribution.
        """

        scores = np.asarray(
            scores
        )

        labels = np.asarray(
            labels
        )

        normal_scores = scores[
            labels == 0
        ]

        fault_scores = scores[
            labels == 1
        ]

        plt.figure(
            figsize=(9, 6)
        )

        plt.hist(
            normal_scores,
            bins=50,
            alpha=0.65,
            label="Normal",
        )

        plt.hist(
            fault_scores,
            bins=50,
            alpha=0.65,
            label="Fault",
        )

        plt.axvline(
            threshold,
            linestyle="--",
            linewidth=2,
            label=(
                f"Threshold = "
                f"{threshold:.4f}"
            ),
        )

        plt.xlabel(
            "Fusion Anomaly Score"
        )

        plt.ylabel(
            "Number of Windows"
        )

        plt.title(
            title
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        print(
            f"✓ Fusion score distribution saved:"
            f"\n  {output_path}"
        )

    # ======================================================
    # INFERENCE PERFORMANCE
    # ======================================================

    def measure_inference_time(
        self,
        isolation_scores,
        autoencoder_errors,
    ):
        """
        Measure inference overhead of the fusion layer.

        This does NOT include:
            - Isolation Forest inference
            - Autoencoder inference
            - Wavelet extraction
            - preprocessing
            - Kalman filtering
            - LSTM inference
        """

        isolation_scores = np.asarray(
            isolation_scores,
            dtype=np.float64,
        ).reshape(-1)

        autoencoder_errors = np.asarray(
            autoencoder_errors,
            dtype=np.float64,
        ).reshape(-1)

        start_time = (
            time.perf_counter()
        )

        (
            isolation_normalized,
            autoencoder_normalized,
        ) = self.transform_scores(
            isolation_scores,
            autoencoder_errors,
        )

        fused_scores = (
            self.fuse_scores(
                isolation_normalized,
                autoencoder_normalized,
            )
        )

        predictions = (
            self.predict(
                fused_scores
            )
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        samples = len(
            isolation_scores
        )

        milliseconds_per_window = (
            elapsed
            / samples
            * 1000
        )

        print("\n" + "=" * 70)
        print(
            "PROPOSED FUSION INFERENCE PERFORMANCE"
        )
        print("=" * 70)

        print(
            f"Total Samples       : "
            f"{samples}"
        )

        print(
            f"Total Inference Time: "
            f"{elapsed:.6f} seconds"
        )

        print(
            f"Time per Window     : "
            f"{milliseconds_per_window:.6f} ms"
        )

        return {
            "total_inference_time_seconds":
                elapsed,

            "milliseconds_per_window":
                milliseconds_per_window,

            "predictions":
                predictions,
        }