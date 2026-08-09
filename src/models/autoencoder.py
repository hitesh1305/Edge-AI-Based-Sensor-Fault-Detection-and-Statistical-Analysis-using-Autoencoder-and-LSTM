"""
Dense Autoencoder for Sensor Anomaly Detection

Research pipeline:

Training
    Healthy training windows
        ↓
    Autoencoder
        ↓
    Reconstruction error

Validation
    Reconstruction errors
        ↓
    Threshold selection

Test
    Fixed threshold
        ↓
    Final anomaly detection metrics
"""

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

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from configs.config import (
    MODELS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
)


class AutoencoderDetector:

    def __init__(
        self,
        input_dim,
        encoding_dim=64,
        learning_rate=0.001,
        batch_size=128,
        epochs=50,
    ):

        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.model = None
        self.threshold = None

    # ======================================================
    # BUILD MODEL
    # ======================================================

    def build_model(self):

        inputs = Input(
            shape=(self.input_dim,)
        )

        # Encoder
        encoded = Dense(
            128,
            activation="relu",
            name="encoder_dense_1"
        )(inputs)

        encoded = Dense(
            self.encoding_dim,
            activation="relu",
            name="latent_representation"
        )(encoded)

        # Decoder
        decoded = Dense(
            128,
            activation="relu",
            name="decoder_dense_1"
        )(encoded)

        outputs = Dense(
            self.input_dim,
            activation="linear",
            name="reconstruction"
        )(decoded)

        self.model = Model(
            inputs,
            outputs
        )

        optimizer = Adam(
            learning_rate=self.learning_rate
        )

        self.model.compile(
            optimizer=optimizer,
            loss="mse"
        )

        print("\n" + "=" * 70)
        print("DENSE AUTOENCODER ARCHITECTURE")
        print("=" * 70)

        self.model.summary()

        return self.model

    # ======================================================
    # TRAIN
    # ======================================================

    def fit(
        self,
        X_train_healthy,
        X_validation_healthy
    ):

        if self.model is None:
            self.build_model()

        print("\n" + "=" * 70)
        print("DENSE AUTOENCODER TRAINING")
        print("=" * 70)

        print(
            f"Healthy Training Samples   : "
            f"{len(X_train_healthy)}"
        )

        print(
            f"Healthy Validation Samples : "
            f"{len(X_validation_healthy)}"
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        )

        start_time = time.perf_counter()

        history = self.model.fit(
            X_train_healthy,
            X_train_healthy,
            validation_data=(
                X_validation_healthy,
                X_validation_healthy,
            ),
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=True,
            callbacks=[
                early_stopping
            ],
            verbose=1,
        )

        training_time = (
            time.perf_counter()
            - start_time
        )

        print(
            f"\nTraining Time : "
            f"{training_time:.4f} seconds"
        )

        return history, training_time

    # ======================================================
    # RECONSTRUCTION ERROR
    # ======================================================

    def reconstruction_errors(
        self,
        X
    ):

        reconstructed = self.model.predict(
            X,
            batch_size=self.batch_size,
            verbose=0,
        )

        errors = np.mean(
            np.square(
                X - reconstructed
            ),
            axis=1
        )

        return errors

    # ======================================================
    # THRESHOLD SELECTION
    # ======================================================

    def select_threshold(
        self,
        validation_errors,
        validation_labels
    ):

        print("\n" + "=" * 70)
        print("AUTOENCODER THRESHOLD SELECTION")
        print("=" * 70)

        candidate_thresholds = np.unique(
            np.percentile(
                validation_errors,
                np.linspace(
                    50,
                    99.9,
                    500
                )
            )
        )

        best_threshold = None
        best_f1 = -1.0

        best_precision = 0.0
        best_recall = 0.0

        for threshold in candidate_thresholds:

            predictions = (
                validation_errors
                >= threshold
            ).astype(int)

            current_f1 = f1_score(
                validation_labels,
                predictions,
                zero_division=0
            )

            if current_f1 > best_f1:

                best_f1 = current_f1

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
    # PREDICT
    # ======================================================

    def predict(
        self,
        X
    ):

        if self.threshold is None:
            raise RuntimeError(
                "Threshold has not been selected."
            )

        errors = self.reconstruction_errors(
            X
        )

        predictions = (
            errors >= self.threshold
        ).astype(int)

        return predictions, errors

    # ======================================================
    # EVALUATION
    # ======================================================

    def evaluate(
        self,
        labels,
        predictions,
        errors,
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

        if len(
            np.unique(labels)
        ) == 2:

            roc_auc = roc_auc_score(
                labels,
                errors
            )

        else:

            roc_auc = np.nan

        matrix = confusion_matrix(
            labels,
            predictions
        )

        results = {
            "model": "Dense Autoencoder",
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
            f"DENSE AUTOENCODER - "
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

        print(matrix)

        return results

    # ======================================================
    # SAVE MODEL
    # ======================================================

    def save_model(self):

        output_dir = (
            MODELS_DIR
            / "autoencoder"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        model_path = (
            output_dir
            / "autoencoder.keras"
        )

        self.model.save(
            model_path
        )

        print(
            f"\n✓ Autoencoder saved:"
            f"\n  {model_path}"
        )

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    @staticmethod
    def save_results(
        results
    ):

        output_dir = (
            TABLES_DIR
            / "autoencoder"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            output_dir
            / "results.csv"
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
    # TRAINING HISTORY PLOT
    # ======================================================

    @staticmethod
    def plot_training_history(
        history
    ):

        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.figure(
            figsize=(9, 6)
        )

        plt.plot(
            history.history["loss"],
            label="Training Loss"
        )

        plt.plot(
            history.history["val_loss"],
            label="Validation Loss"
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "MSE Loss"
        )

        plt.title(
            "Autoencoder Training History"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        output_file = (
            FIGURES_DIR
            / "autoencoder_training_history.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ======================================================
    # RECONSTRUCTION ERROR PLOT
    # ======================================================

    @staticmethod
    def plot_error_distribution(
        errors,
        labels,
        threshold,
        split_name
    ):

        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        normal_errors = errors[
            labels == 0
        ]

        anomaly_errors = errors[
            labels == 1
        ]

        plt.figure(
            figsize=(9, 6)
        )

        plt.hist(
            normal_errors,
            bins=50,
            alpha=0.6,
            label="Normal"
        )

        plt.hist(
            anomaly_errors,
            bins=50,
            alpha=0.6,
            label="Degraded"
        )

        plt.axvline(
            threshold,
            linestyle="--",
            linewidth=2,
            label="Decision Threshold"
        )

        plt.xlabel(
            "Reconstruction Error"
        )

        plt.ylabel(
            "Number of Windows"
        )

        plt.title(
            f"Autoencoder Reconstruction "
            f"Error - {split_name.capitalize()}"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        output_file = (
            FIGURES_DIR
            / f"autoencoder_reconstruction_"
              f"error_{split_name}.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ======================================================
    # CONFUSION MATRIX
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
            f"Autoencoder Confusion Matrix - "
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
            ["Normal", "Degraded"]
        )

        plt.yticks(
            [0, 1],
            ["Normal", "Degraded"]
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
            / f"autoencoder_confusion_"
              f"{split_name}.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ======================================================
    # ROC CURVE
    # ======================================================

    @staticmethod
    def plot_roc_curve(
        labels,
        errors,
        split_name
    ):

        if len(
            np.unique(labels)
        ) < 2:

            return

        fpr, tpr, _ = roc_curve(
            labels,
            errors
        )

        auc = roc_auc_score(
            labels,
            errors
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
            f"Autoencoder ROC Curve - "
            f"{split_name.capitalize()}"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        output_file = (
            FIGURES_DIR
            / f"autoencoder_roc_"
              f"{split_name}.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # ======================================================
    # INFERENCE TIME
    # ======================================================

    def measure_inference_time(
        self,
        X
    ):

        start_time = time.perf_counter()

        self.model.predict(
            X,
            batch_size=self.batch_size,
            verbose=0
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
            "AUTOENCODER INFERENCE PERFORMANCE"
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