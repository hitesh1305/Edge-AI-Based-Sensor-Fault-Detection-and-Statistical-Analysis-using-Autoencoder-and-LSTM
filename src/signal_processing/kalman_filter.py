"""
Kalman Filter Based State Estimation

Applies a 1-D Kalman Filter independently to every
sensor of every engine.

The implementation:
- Processes each engine independently.
- Processes each sensor independently.
- Preserves all non-sensor columns.
- Saves filtered datasets.
- Generates valid before/after plots.
- Automatically selects an engine that actually exists
  in the current dataset.
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
    KF_INITIAL_ERROR,
)


class KalmanFilterEstimator:

    def __init__(
        self,
        process_variance=KF_PROCESS_VARIANCE,
        measurement_variance=KF_MEASUREMENT_VARIANCE,
        initial_error=KF_INITIAL_ERROR,
    ):

        self.Q = process_variance
        self.R = measurement_variance
        self.P0 = initial_error

    # ==========================================================
    # 1-D Kalman Filter
    # ==========================================================

    def _filter_signal(self, signal):

        if len(signal) == 0:
            return []

        x_est = float(signal[0])
        P = self.P0

        filtered = []

        for measurement in signal:

            measurement = float(measurement)

            # --------------------------------------------------
            # Prediction
            # --------------------------------------------------

            x_pred = x_est

            P_pred = P + self.Q

            # --------------------------------------------------
            # Kalman Gain
            # --------------------------------------------------

            denominator = P_pred + self.R

            if denominator <= 0:
                K = 0.0
            else:
                K = P_pred / denominator

            # --------------------------------------------------
            # Measurement Update
            # --------------------------------------------------

            x_est = (
                x_pred
                + K * (measurement - x_pred)
            )

            # --------------------------------------------------
            # Covariance Update
            # --------------------------------------------------

            P = (1.0 - K) * P_pred

            filtered.append(x_est)

        return filtered

    # ==========================================================
    # Apply Kalman Filter
    # ==========================================================

    def apply(
        self,
        df,
        split_name="dataset"
    ):

        print("\n" + "=" * 70)
        print("KALMAN FILTER STATE ESTIMATION")
        print("=" * 70)

        filtered_df = df.copy()

        # ------------------------------------------------------
        # Identify sensor columns
        # ------------------------------------------------------

        sensor_columns = [
            col
            for col in filtered_df.columns
            if col.startswith("sensor_")
        ]

        engines = sorted(
            filtered_df["engine_id"]
            .unique()
        )

        print(
            f"Engines : {len(engines)}"
        )

        print(
            f"Sensors : {len(sensor_columns)}"
        )

        print(
            f"Split   : {split_name}"
        )

        # ------------------------------------------------------
        # Filter each engine independently
        # ------------------------------------------------------

        for engine in engines:

            engine_mask = (
                filtered_df["engine_id"]
                == engine
            )

            engine_data = (
                filtered_df.loc[
                    engine_mask
                ]
                .sort_values("cycle")
            )

            for sensor in sensor_columns:

                signal = (
                    engine_data[sensor]
                    .to_numpy()
                )

                filtered_signal = (
                    self._filter_signal(
                        signal
                    )
                )

                filtered_df.loc[
                    engine_data.index,
                    sensor
                ] = filtered_signal

        # ------------------------------------------------------
        # Save filtered dataset
        # ------------------------------------------------------

        TABLES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            TABLES_DIR
            / f"kalman_filtered_{split_name}.csv"
        )

        filtered_df.to_csv(
            output_file,
            index=False
        )

        print(
            f"\n✓ Filtered dataset saved:"
            f"\n  {output_file}"
        )

        # ------------------------------------------------------
        # Generate comparison plots
        # ------------------------------------------------------

        self._plot_examples(
            original=df,
            filtered=filtered_df,
            sensor_columns=sensor_columns,
            split_name=split_name,
        )

        print(
            "✓ Comparison plots saved"
        )

        return filtered_df

    # ==========================================================
    # Plot Examples
    # ==========================================================

    def _plot_examples(
        self,
        original,
        filtered,
        sensor_columns,
        split_name,
    ):

        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # ------------------------------------------------------
        # Automatically select the first available engine
        # ------------------------------------------------------

        engines = sorted(
            original["engine_id"]
            .unique()
        )

        if len(engines) == 0:
            print(
                "⚠ No engines available for plotting."
            )
            return

        engine = engines[0]

        original_engine = original[
            original["engine_id"] == engine
        ].sort_values("cycle")

        filtered_engine = filtered[
            filtered["engine_id"] == engine
        ].sort_values("cycle")

        if original_engine.empty:
            print(
                "⚠ No data available for plot."
            )
            return

        # ------------------------------------------------------
        # Representative sensors
        # ------------------------------------------------------

        preferred_sensors = [
            "sensor_2",
            "sensor_9",
            "sensor_14",
        ]

        sensors = [
            sensor
            for sensor in preferred_sensors
            if sensor in sensor_columns
        ]

        # Fallback if one of the preferred sensors
        # does not exist.
        if not sensors:
            sensors = sensor_columns[:3]

        # ------------------------------------------------------
        # Generate plots
        # ------------------------------------------------------

        for sensor in sensors:

            plt.figure(
                figsize=(10, 5)
            )

            plt.plot(
                original_engine["cycle"],
                original_engine[sensor],
                label="Original",
                alpha=0.65,
            )

            plt.plot(
                filtered_engine["cycle"],
                filtered_engine[sensor],
                label="Kalman Filtered",
                linewidth=2,
            )

            plt.title(
                f"{sensor} - Engine {engine} "
                f"({split_name})"
            )

            plt.xlabel(
                "Cycle"
            )

            plt.ylabel(
                sensor
            )

            plt.legend()

            plt.grid(
                alpha=0.3
            )

            plt.tight_layout()

            output_file = (
                FIGURES_DIR
                / f"{sensor}_kalman_{split_name}.png"
            )

            plt.savefig(
                output_file,
                dpi=FIG_DPI,
                bbox_inches="tight",
            )

            plt.close()

        print(
            f"  Plots generated using "
            f"Engine {engine} ({split_name})"
        )