"""
LSTM-Based Remaining Useful Life (RUL) Predictor

Input:
    Sequential sensor windows
    Shape: (samples, timesteps, features)

Output:
    Continuous RUL prediction

Evaluation:
    MAE
    RMSE
    R2
    Training/validation loss
    Predicted vs actual RUL
    RUL error distribution
    Inference time
"""

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from configs.config import (
    MODELS_DIR,
    FIGURES_DIR,
    TABLES_DIR,
)


class LSTMPredictor:

    def __init__(
        self,
        timesteps,
        features,
        lstm_units=64,
        learning_rate=0.001,
        batch_size=128,
        epochs=50,
        dropout=0.2,
    ):

        self.timesteps = timesteps
        self.features = features
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout = dropout

        self.model = None

    # ======================================================
    # BUILD MODEL
    # ======================================================

    def build_model(self):

        self.model = Sequential(
            [
                Input(
                    shape=(
                        self.timesteps,
                        self.features,
                    )
                ),

                LSTM(
                    self.lstm_units,
                    return_sequences=True,
                    name="lstm_1",
                ),

                Dropout(
                    self.dropout,
                    name="dropout_1",
                ),

                LSTM(
                    self.lstm_units // 2,
                    return_sequences=False,
                    name="lstm_2",
                ),

                Dropout(
                    self.dropout,
                    name="dropout_2",
                ),

                Dense(
                    32,
                    activation="relu",
                    name="dense_1",
                ),

                Dense(
                    1,
                    activation="linear",
                    name="rul_output",
                ),
            ],
            name="lstm_rul_predictor",
        )

        optimizer = Adam(
            learning_rate=self.learning_rate
        )

        self.model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae"],
        )

        print("\n" + "=" * 70)
        print("LSTM RUL PREDICTOR ARCHITECTURE")
        print("=" * 70)

        self.model.summary()

        return self.model

    # ======================================================
    # TRAIN
    # ======================================================

    def fit(
        self,
        X_train,
        y_train,
        X_validation,
        y_validation,
    ):

        if self.model is None:
            self.build_model()

        print("\n" + "=" * 70)
        print("LSTM RUL PREDICTOR TRAINING")
        print("=" * 70)

        print(
            f"Training Samples   : {len(X_train)}"
        )

        print(
            f"Validation Samples : {len(X_validation)}"
        )

        print(
            f"Input Timesteps    : {self.timesteps}"
        )

        print(
            f"Sensor Features    : {self.features}"
        )

        print(
            f"LSTM Units         : {self.lstm_units}"
        )

        print(
            f"Learning Rate      : {self.learning_rate}"
        )

        print(
            f"Batch Size         : {self.batch_size}"
        )

        print(
            f"Maximum Epochs     : {self.epochs}"
        )

        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        )

        start_time = time.perf_counter()

        history = self.model.fit(
            X_train,
            y_train,
            validation_data=(
                X_validation,
                y_validation,
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
    # PREDICTION
    # ======================================================

    def predict(
        self,
        X,
    ):

        predictions = self.model.predict(
            X,
            batch_size=self.batch_size,
            verbose=0,
        )

        return predictions.reshape(-1)

    # ======================================================
    # EVALUATION
    # ======================================================

    def evaluate(
        self,
        y_true,
        y_pred,
        split_name,
    ):

        mae = mean_absolute_error(
            y_true,
            y_pred,
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )

        r2 = r2_score(
            y_true,
            y_pred,
        )

        errors = (
            y_pred - y_true
        )

        mean_error = np.mean(
            errors
        )

        std_error = np.std(
            errors
        )

        results = {
            "model": "LSTM RUL Predictor",
            "split": split_name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mean_error": mean_error,
            "std_error": std_error,
            "samples": len(y_true),
        }

        print("\n" + "-" * 70)

        print(
            f"LSTM RUL PREDICTION - "
            f"{split_name.upper()}"
        )

        print("-" * 70)

        print(
            f"MAE  : {mae:.4f}"
        )

        print(
            f"RMSE : {rmse:.4f}"
        )

        print(
            f"R²   : {r2:.4f}"
        )

        print(
            f"Mean Error : {mean_error:.4f}"
        )

        print(
            f"Std Error  : {std_error:.4f}"
        )

        return results

    # ======================================================
    # SAVE MODEL
    # ======================================================

    def save_model(self):

        output_dir = (
            MODELS_DIR
            / "lstm_predictor"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        model_path = (
            output_dir
            / "lstm_rul_predictor.keras"
        )

        self.model.save(
            model_path
        )

        print(
            f"\n✓ LSTM RUL Predictor saved:"
            f"\n  {model_path}"
        )

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    @staticmethod
    def save_results(
        results,
    ):

        output_dir = (
            TABLES_DIR
            / "lstm_predictor"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
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
                    new_results,
                ],
                ignore_index=True,
            )

        new_results.to_csv(
            output_file,
            index=False,
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
        history,
    ):

        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure(
            figsize=(9, 6)
        )

        plt.plot(
            history.history["loss"],
            label="Training Loss",
        )

        plt.plot(
            history.history["val_loss"],
            label="Validation Loss",
        )

        plt.xlabel(
            "Epoch"
        )

        plt.ylabel(
            "MSE Loss"
        )

        plt.title(
            "LSTM RUL Predictor Training History"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        output_file = (
            FIGURES_DIR
            / "lstm_rul_training_history.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    # ======================================================
    # PREDICTED VS ACTUAL RUL
    # ======================================================

    @staticmethod
    def plot_predictions(
        y_true,
        y_pred,
        split_name,
        max_samples=500,
    ):

        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        sample_count = min(
            len(y_true),
            max_samples,
        )

        x = np.arange(
            sample_count
        )

        plt.figure(
            figsize=(12, 6)
        )

        plt.plot(
            x,
            y_true[:sample_count],
            label="Actual RUL",
            linewidth=2,
        )

        plt.plot(
            x,
            y_pred[:sample_count],
            label="Predicted RUL",
            linewidth=1.5,
        )

        plt.xlabel(
            "Test Window"
        )

        plt.ylabel(
            "Remaining Useful Life"
        )

        plt.title(
            f"LSTM RUL Prediction - "
            f"{split_name.capitalize()}"
        )

        plt.legend()

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        output_file = (
            FIGURES_DIR
            / f"lstm_rul_prediction_"
              f"{split_name}.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    # ======================================================
    # ERROR DISTRIBUTION
    # ======================================================

    @staticmethod
    def plot_error_distribution(
        y_true,
        y_pred,
        split_name,
    ):

        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        errors = (
            y_pred - y_true
        )

        plt.figure(
            figsize=(9, 6)
        )

        plt.hist(
            errors,
            bins=50,
            alpha=0.75,
        )

        plt.axvline(
            0,
            linestyle="--",
            linewidth=2,
        )

        plt.xlabel(
            "Prediction Error"
        )

        plt.ylabel(
            "Number of Windows"
        )

        plt.title(
            f"LSTM RUL Prediction Error "
            f"Distribution - "
            f"{split_name.capitalize()}"
        )

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        output_file = (
            FIGURES_DIR
            / f"lstm_rul_error_distribution_"
              f"{split_name}.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    # ======================================================
    # SCATTER: ACTUAL VS PREDICTED
    # ======================================================

    @staticmethod
    def plot_actual_vs_predicted(
        y_true,
        y_pred,
        split_name,
    ):

        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.figure(
            figsize=(7, 7)
        )

        plt.scatter(
            y_true,
            y_pred,
            alpha=0.35,
            s=15,
        )

        minimum = min(
            np.min(y_true),
            np.min(y_pred),
        )

        maximum = max(
            np.max(y_true),
            np.max(y_pred),
        )

        plt.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
            linewidth=2,
        )

        plt.xlabel(
            "Actual RUL"
        )

        plt.ylabel(
            "Predicted RUL"
        )

        plt.title(
            f"Actual vs Predicted RUL - "
            f"{split_name.capitalize()}"
        )

        plt.grid(
            alpha=0.3
        )

        plt.tight_layout()

        output_file = (
            FIGURES_DIR
            / f"lstm_rul_actual_vs_predicted_"
              f"{split_name}.png"
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    # ======================================================
    # INFERENCE PERFORMANCE
    # ======================================================

    def measure_inference_time(
        self,
        X,
    ):

        start_time = time.perf_counter()

        self.model.predict(
            X,
            batch_size=self.batch_size,
            verbose=0,
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        samples = len(X)

        milliseconds_per_window = (
            elapsed
            / samples
            * 1000
        )

        print("\n" + "-" * 70)

        print(
            "LSTM RUL PREDICTOR "
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
            f"{milliseconds_per_window:.6f} ms"
        )

        return {
            "total_inference_time_seconds":
                elapsed,
            "milliseconds_per_window":
                milliseconds_per_window,
        }