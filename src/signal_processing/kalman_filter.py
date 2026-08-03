"""
Kalman Filter Based State Estimation

Applies a 1-D Kalman Filter independently to
each sensor of every engine.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from configs.config import (
    FIGURES_DIR,
    TABLES_DIR,
    FIG_DPI,
    KF_PROCESS_VARIANCE,
    KF_MEASUREMENT_VARIANCE,
    KF_INITIAL_ERROR
)


class KalmanFilterEstimator:

    def __init__(
        self,
        process_variance=KF_PROCESS_VARIANCE,
        measurement_variance=KF_MEASUREMENT_VARIANCE,
        initial_error=KF_INITIAL_ERROR
    ):

        self.Q = process_variance
        self.R = measurement_variance
        self.P0 = initial_error

    def _filter_signal(self, signal):

        x_est = signal[0]
        P = self.P0

        filtered = []

        for measurement in signal:

            # Prediction
            x_pred = x_est
            P_pred = P + self.Q

            # Update
            K = P_pred / (P_pred + self.R)

            x_est = x_pred + K * (measurement - x_pred)

            P = (1 - K) * P_pred

            filtered.append(x_est)

        return filtered

    def apply(self, df):

        print("\n" + "=" * 70)
        print("KALMAN FILTER STATE ESTIMATION")
        print("=" * 70)

        filtered_df = df.copy()

        sensor_columns = [
            col
            for col in filtered_df.columns
            if col.startswith("sensor_")
        ]

        engines = filtered_df["engine_id"].unique()

        print(f"Engines : {len(engines)}")
        print(f"Sensors : {len(sensor_columns)}")

        for engine in engines:

            mask = filtered_df["engine_id"] == engine

            engine_data = filtered_df.loc[mask]

            for sensor in sensor_columns:

                filtered_signal = self._filter_signal(
                    engine_data[sensor].values
                )

                filtered_df.loc[
                    mask,
                    sensor
                ] = filtered_signal

        TABLES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        filtered_df.to_csv(
            TABLES_DIR / "kalman_filtered.csv",
            index=False
        )

        print("\n✓ Filtered dataset saved")

        self._plot_examples(df, filtered_df)

        print("✓ Comparison plots saved")

        return filtered_df

    def _plot_examples(
        self,
        original,
        filtered
    ):

        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        sensors = [
            "sensor_2",
            "sensor_9",
            "sensor_14"
        ]

        engine = 1

        mask = original["engine_id"] == engine

        for sensor in sensors:

            plt.figure(figsize=(10,4))

            plt.plot(
                original.loc[mask, "cycle"],
                original.loc[mask, sensor],
                label="Original",
                alpha=0.7
            )

            plt.plot(
                filtered.loc[mask, "cycle"],
                filtered.loc[mask, sensor],
                label="Kalman",
                linewidth=2
            )

            plt.title(f"{sensor} - Engine {engine}")

            plt.xlabel("Cycle")

            plt.ylabel(sensor)

            plt.legend()

            plt.grid(alpha=0.3)

            plt.tight_layout()

            plt.savefig(
                FIGURES_DIR / f"{sensor}_kalman.png",
                dpi=FIG_DPI
            )

            plt.close()