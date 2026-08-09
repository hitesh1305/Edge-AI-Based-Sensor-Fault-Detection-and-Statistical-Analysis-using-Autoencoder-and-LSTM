"""
Fault Score Fusion
==================

Research-grade fusion of:
    1. Isolation Forest anomaly score
    2. Dense Autoencoder reconstruction error

The module:
    - normalizes both anomaly scores
    - learns the fusion threshold using validation data only
    - evaluates the final fused detector on the test set
    - reports Accuracy, Precision, Recall, F1 and ROC-AUC
    - generates confusion matrix, ROC and score-distribution figures
    - measures inference time
    - saves raw/fused scores for reproducibility

Important:
    Validation data is used for threshold selection.
    Test data is used only for final evaluation.
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

from sklearn.preprocessing import MinMaxScaler


class FaultScoreFusion:

    def __init__(
        self,
        isolation_weight=0.5,
        autoencoder_weight=0.5,
    ):
        """
        Parameters
        ----------
        isolation_weight : float
            Weight assigned to Isolation Forest anomaly score.

        autoencoder_weight : float
            Weight assigned to Autoencoder reconstruction error.

        The weights are normalized automatically.
        """

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

        self.scaler = None
        self.threshold = None

    # ======================================================
    # SCORE PREPARATION
    # ======================================================

    def fit_scaler(
        self,
        isolation_scores,
        autoencoder_errors,
    ):
        """
        Fit score normalization using VALIDATION data only.

        This prevents test-set information leakage.
        """

        isolation_scores = np.asarray(
            isolation_scores,
            dtype=np.float64,
        ).reshape(-1, 1)

        autoencoder_errors = np.asarray(
            autoencoder_errors,
            dtype=np.float64,
        ).reshape(-1, 1)

        combined = np.hstack(
            [
                isolation_scores,
                autoencoder_errors,
            ]
        )

        self.scaler = MinMaxScaler()

        self.scaler.fit(combined)

        return self

    def transform_scores(
        self,
        isolation_scores,
        autoencoder_errors,
    ):
        """
        Normalize Isolation Forest and Autoencoder scores
        using the scaler fitted on validation data.
        """

        if self.scaler is None:
            raise RuntimeError(
                "Scaler has not been fitted. "
                "Call fit_scaler() first."
            )

        isolation_scores = np.asarray(
            isolation_scores,
            dtype=np.float64,
        ).reshape(-1, 1)

        autoencoder_errors = np.asarray(
            autoencoder_errors,
            dtype=np.float64,
        ).reshape(-1, 1)

        combined = np.hstack(
            [
                isolation_scores,
                autoencoder_errors,
            ]
        )

        normalized = (
            self.scaler.transform(
                combined
            )
        )

        isolation_normalized = (
            normalized[:, 0]
        )

        autoencoder_normalized = (
            normalized[:, 1]
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
        )

        autoencoder_errors = np.asarray(
            autoencoder_errors,
            dtype=np.float64,
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
    # THRESHOLD SELECTION
    # ======================================================

    def select_threshold(
        self,
        fused_validation_scores,
        validation_labels,
    ):
        """
        Select threshold using validation F1.

        The test set is NOT used here.
        """

        scores = np.asarray(
            fused_validation_scores,
            dtype=np.float64,
        )

        labels = np.asarray(
            validation_labels,
            dtype=int,
        )

        unique_scores = np.unique(
            scores
        )

        if len(unique_scores) > 1000:

            thresholds = np.percentile(
                scores,
                np.linspace(
                    0.0,
                    100.0,
                    1001,
                ),
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

            current_precision = (
                precision_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            )

            current_recall = (
                recall_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            )

            if current_f1 > best_f1:

                best_f1 = current_f1

                best_threshold = (
                    threshold
                )

                best_precision = (
                    current_precision
                )

                best_recall = (
                    current_recall
                )

        self.threshold = (
            float(best_threshold)
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

        print("\n" + "=" * 70)
        print(
            f"PROPOSED FUSION - "
            f"{split_name.upper()}"
        )
        print("=" * 70)

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

        print("\nConfusion Matrix:")

        print(cm)

        results = {
            "model": "Proposed Score Fusion",
            "split": split_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "threshold": self.threshold,
            "samples": len(labels),
            "actual_anomalies": int(
                np.sum(labels == 1)
            ),
            "predicted_anomalies": int(
                np.sum(predictions == 1)
            ),
            "isolation_weight":
                self.isolation_weight,
            "autoencoder_weight":
                self.autoencoder_weight,
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

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = pd.DataFrame(
            {
                "window_id": window_ids,
                "actual_label": labels,
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

        plt.title(title)

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

        threshold = (
            cm.max() / 2.0
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
            label=f"Fusion ROC-AUC = {auc:.4f}",
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

        isolation_scores = np.asarray(
            isolation_scores,
            dtype=np.float64,
        )

        autoencoder_errors = np.asarray(
            autoencoder_errors,
            dtype=np.float64,
        )

        start_time = (
            time.perf_counter()
        )

        isolation_normalized, autoencoder_normalized = (
            self.transform_scores(
                isolation_scores,
                autoencoder_errors,
            )
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
        print("PROPOSED FUSION INFERENCE PERFORMANCE")
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