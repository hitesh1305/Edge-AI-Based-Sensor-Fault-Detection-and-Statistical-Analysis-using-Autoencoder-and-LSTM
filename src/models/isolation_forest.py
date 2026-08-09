"""
Isolation Forest Anomaly Detection

Research-grade implementation for the NASA C-MAPSS pipeline.

Workflow:
    Train       -> fit Isolation Forest
    Validation  -> select anomaly threshold
    Test        -> final evaluation

The test set is never used for model fitting or threshold tuning.
"""

from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

from configs.config import (
    MODELS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
)


class IsolationForestDetector:

    def __init__(
        self,
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    ):

        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        self.threshold = None

        self.feature_columns = None

    # ======================================================
    # Prepare Features
    # ======================================================

    @staticmethod
    def prepare_features(df):

        feature_columns = [
            column
            for column in df.columns
            if column != "window_id"
        ]

        X = df[
            feature_columns
        ].replace(
            [np.inf, -np.inf],
            np.nan
        )

        X = X.fillna(
            X.median()
        )

        return X, feature_columns

    # ======================================================
    # Ground-Truth Label
    # ======================================================

    @staticmethod
    def create_rul_labels(
        metadata,
        rul_threshold=30
    ):
        """
        Creates evaluation labels from RUL.

        Label definition:

            0 -> healthy / non-critical
            1 -> degraded / critical

        A window is considered degraded when its target
        RUL is <= rul_threshold.

        This label is used ONLY for evaluation.

        It is NOT provided to the Isolation Forest.
        """

        if "target_rul" not in metadata.columns:
            raise ValueError(
                "metadata must contain "
                "'target_rul'."
            )

        labels = (
            metadata["target_rul"]
            <= rul_threshold
        ).astype(int)

        return labels.to_numpy()

    # ======================================================
    # Train
    # ======================================================

    def fit(
        self,
        wavelet_train
    ):

        print("\n" + "=" * 70)
        print("ISOLATION FOREST TRAINING")
        print("=" * 70)

        X_train, feature_columns = (
            self.prepare_features(
                wavelet_train
            )
        )

        self.feature_columns = (
            feature_columns
        )

        print(
            f"Training Samples : "
            f"{X_train.shape[0]}"
        )

        print(
            f"Features         : "
            f"{X_train.shape[1]}"
        )

        start_time = time.perf_counter()

        self.model.fit(
            X_train
        )

        training_time = (
            time.perf_counter()
            - start_time
        )

        print(
            f"Training Time    : "
            f"{training_time:.4f} seconds"
        )

        # Save trained model
        MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = (
            MODELS_DIR
            / "isolation_forest.pkl"
        )

        joblib.dump(
            self.model,
            model_path
        )

        print(
            f"✓ Model saved:"
            f"\n  {model_path}"
        )

        return training_time

    # ======================================================
    # Anomaly Score
    # ======================================================

    def anomaly_scores(
        self,
        wavelet_df
    ):
        """
        Isolation Forest's decision_function:
            higher -> more normal

        We convert it to:
            higher -> more anomalous
        """

        X, _ = self.prepare_features(
            wavelet_df
        )

        scores = -self.model.decision_function(
            X
        )

        return scores

    # ======================================================
    # Threshold Selection
    # ======================================================

    def select_threshold(
        self,
        validation_scores,
        validation_labels
    ):
        """
        Select the threshold using validation data only.

        The selected threshold maximizes F1 score.
        """

        print("\n" + "=" * 70)
        print("ISOLATION FOREST THRESHOLD SELECTION")
        print("=" * 70)

        unique_scores = np.sort(
            np.unique(
                validation_scores
            )
        )

        best_threshold = None
        best_f1 = -1.0

        best_precision = 0.0
        best_recall = 0.0

        for threshold in unique_scores:

            predictions = (
                validation_scores
                >= threshold
            ).astype(int)

            f1 = f1_score(
                validation_labels,
                predictions,
                zero_division=0
            )

            if f1 > best_f1:

                best_f1 = f1

                best_threshold = (
                    threshold
                )

                best_precision = (
                    precision_score(
                        validation_labels,
                        predictions,
                        zero_division=0
                    )
                )

                best_recall = (
                    recall_score(
                        validation_labels,
                        predictions,
                        zero_division=0
                    )
                )

        self.threshold = (
            best_threshold
        )

        print(
            f"Selected Threshold : "
            f"{self.threshold:.6f}"
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
    # Prediction
    # ======================================================

    def predict(
        self,
        wavelet_df
    ):

        if self.threshold is None:
            raise RuntimeError(
                "Threshold has not been selected."
            )

        scores = self.anomaly_scores(
            wavelet_df
        )

        predictions = (
            scores >= self.threshold
        ).astype(int)

        return predictions, scores

    # ======================================================
    # Evaluation
    # ======================================================

    def evaluate(
        self,
        labels,
        predictions,
        scores,
        split_name
    ):

        accuracy = accuracy_score(
            labels,
            predictions
        )

        precision = precision_score(
            labels,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            labels,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            labels,
            predictions,
            zero_division=0
        )

        # ROC-AUC requires both classes.
        if len(np.unique(labels)) == 2:
            roc_auc = roc_auc_score(
                labels,
                scores
            )
        else:
            roc_auc = np.nan

        matrix = confusion_matrix(
            labels,
            predictions
        )

        results = {
            "model": "Isolation Forest",
            "split": split_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "threshold": self.threshold,
            "samples": len(labels),
            "actual_anomalies": int(
                np.sum(labels)
            ),
            "predicted_anomalies": int(
                np.sum(predictions)
            ),
        }

        print("\n" + "-" * 70)
        print(
            f"ISOLATION FOREST - "
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

        print("\nConfusion Matrix:")

        print(matrix)

        return results

    # ======================================================
    # Save Results
    # ======================================================

    @staticmethod
    def save_results(
        results
    ):

        TABLES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            TABLES_DIR
            / "isolation_forest_results.csv"
        )

        new_results = pd.DataFrame(
            [results]
        )

        if output_file.exists():

            old_results = pd.read_csv(
                output_file
            )

            new_results = pd.concat(
                [
                    old_results,
                    new_results
                ],
                ignore_index=True
            )

        new_results.to_csv(
            output_file,
            index=False
        )

        print(
            f"\n✓ Results saved:"
            f"\n  {output_file}"
        )

    # ======================================================
    # Confusion Matrix Plot
    # ======================================================

    @staticmethod
    def plot_confusion_matrix(
        labels,
        predictions,
        split_name
    ):

        matrix = confusion_matrix(
            labels,
            predictions
        )

        plt.figure(
            figsize=(6, 5)
        )

        plt.imshow(
            matrix
        )

        plt.title(
            f"Isolation Forest "
            f"Confusion Matrix - "
            f"{split_name.capitalize()}"
        )

        plt.xlabel(
            "Predicted Label"
        )

        plt.ylabel(
            "Actual Label"
        )

        plt.xticks(
            [0, 1],
            ["Normal", "Anomaly"]
        )

        plt.yticks(
            [0, 1],
            ["Normal", "Anomaly"]
        )

        for i in range(2):

            for j in range(2):

                plt.text(
                    j,
                    i,
                    matrix[i, j],
                    ha="center",
                    va="center"
                )

        plt.tight_layout()

        output_file = (
            FIGURES_DIR
            / f"isolation_forest_confusion_"
              f"{split_name}.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ======================================================
    # ROC Curve
    # ======================================================

    @staticmethod
    def plot_roc_curve(
        labels,
        scores,
        split_name
    ):

        if len(
            np.unique(labels)
        ) < 2:

            print(
                "⚠ ROC curve skipped: "
                "only one class present."
            )

            return

        fpr, tpr, _ = roc_curve(
            labels,
            scores
        )

        auc = roc_auc_score(
            labels,
            scores
        )

        plt.figure(
            figsize=(7, 6)
        )

        plt.plot(
            fpr,
            tpr,
            label=f"ROC-AUC = {auc:.4f}"
        )

        plt.plot(
            [0, 1],
            [0, 1],
            linestyle="--"
        )

        plt.xlabel(
            "False Positive Rate"
        )

        plt.ylabel(
            "True Positive Rate"
        )

        plt.title(
            f"Isolation Forest ROC Curve - "
            f"{split_name.capitalize()}"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        output_file = (
            FIGURES_DIR
            / f"isolation_forest_roc_"
              f"{split_name}.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ======================================================
    # Anomaly Score Distribution
    # ======================================================

    @staticmethod
    def plot_score_distribution(
        scores,
        labels,
        split_name,
        threshold
    ):

        plt.figure(
            figsize=(9, 6)
        )

        normal_scores = scores[
            labels == 0
        ]

        anomaly_scores = scores[
            labels == 1
        ]

        plt.hist(
            normal_scores,
            bins=40,
            alpha=0.6,
            label="Normal"
        )

        plt.hist(
            anomaly_scores,
            bins=40,
            alpha=0.6,
            label="Anomaly"
        )

        plt.axvline(
            threshold,
            linestyle="--",
            linewidth=2,
            label="Decision Threshold"
        )

        plt.xlabel(
            "Isolation Forest Anomaly Score"
        )

        plt.ylabel(
            "Number of Windows"
        )

        plt.title(
            f"Isolation Forest Anomaly "
            f"Score Distribution - "
            f"{split_name.capitalize()}"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        output_file = (
            FIGURES_DIR
            / f"isolation_forest_score_"
              f"{split_name}.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ======================================================
    # Inference Time
    # ======================================================

    def measure_inference_time(
        self,
        wavelet_df
    ):

        X, _ = self.prepare_features(
            wavelet_df
        )

        start_time = time.perf_counter()

        self.model.decision_function(
            X
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        samples = len(X)

        milliseconds_per_sample = (
            elapsed
            / samples
            * 1000
        )

        print("\n" + "-" * 70)

        print(
            "INFERENCE PERFORMANCE"
        )

        print("-" * 70)

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
            f"{milliseconds_per_sample:.6f} ms"
        )

        return {
            "total_inference_time_seconds":
                elapsed,
            "milliseconds_per_window":
                milliseconds_per_sample,
        }