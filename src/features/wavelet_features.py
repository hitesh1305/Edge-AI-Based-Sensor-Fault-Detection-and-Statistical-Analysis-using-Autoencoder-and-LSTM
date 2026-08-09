"""
Wavelet-Based Feature Extraction

Extracts time-domain and wavelet-domain features from
sliding-window sensor sequences.

Train, validation, and test outputs are saved separately
to prevent accidental overwriting.
"""

import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt

from configs.config import (
    WAVELET_NAME,
    WAVELET_LEVEL,
    FIGURES_DIR,
    TABLES_DIR,
)


class WaveletFeatureExtractor:

    def __init__(
        self,
        wavelet=WAVELET_NAME,
        level=WAVELET_LEVEL
    ):
        self.wavelet = wavelet
        self.level = level
        self._last_X = None

    # ======================================================
    # Statistical Features
    # ======================================================

    @staticmethod
    def _statistical_features(signal):

        signal = np.asarray(
            signal,
            dtype=np.float64
        )

        return {
            "mean": np.mean(signal),
            "std": np.std(signal),
            "variance": np.var(signal),
            "rms": np.sqrt(
                np.mean(signal ** 2)
            ),
            "max": np.max(signal),
            "min": np.min(signal),
            "peak_to_peak": (
                np.max(signal)
                - np.min(signal)
            ),
        }

    # ======================================================
    # Wavelet Energy
    # ======================================================

    @staticmethod
    def _wavelet_energy(coefficients):

        energy = 0.0

        for coefficient in coefficients:

            energy += np.sum(
                np.square(coefficient)
            )

        return energy

    # ======================================================
    # Wavelet Entropy
    # ======================================================

    @staticmethod
    def _wavelet_entropy(coefficients):

        energies = []

        for coefficient in coefficients:

            energy = np.square(
                coefficient
            )

            energies.extend(
                energy.tolist()
            )

        energies = np.asarray(
            energies,
            dtype=np.float64
        )

        total_energy = np.sum(
            energies
        )

        if total_energy <= 0:
            return 0.0

        probabilities = (
            energies / total_energy
        )

        probabilities = probabilities[
            probabilities > 0
        ]

        entropy = -np.sum(
            probabilities
            * np.log2(probabilities)
        )

        return entropy

    # ======================================================
    # Single Sensor
    # ======================================================

    def _extract_sensor_features(
        self,
        signal,
        prefix
    ):

        features = {}

        # --------------------------------------------------
        # Time-domain features
        # --------------------------------------------------

        statistics = (
            self._statistical_features(
                signal
            )
        )

        for name, value in statistics.items():

            features[
                f"{prefix}_{name}"
            ] = value

        # --------------------------------------------------
        # Wavelet decomposition
        # --------------------------------------------------

        wavelet = pywt.Wavelet(
            self.wavelet
        )

        maximum_level = (
            pywt.dwt_max_level(
                data_len=len(signal),
                filter_len=wavelet.dec_len
            )
        )

        level = min(
            self.level,
            maximum_level
        )

        coefficients = pywt.wavedec(
            signal,
            self.wavelet,
            level=level,
            mode="symmetric"
        )

        # --------------------------------------------------
        # Wavelet-domain features
        # --------------------------------------------------

        features[
            f"{prefix}_wavelet_energy"
        ] = self._wavelet_energy(
            coefficients
        )

        features[
            f"{prefix}_wavelet_entropy"
        ] = self._wavelet_entropy(
            coefficients
        )

        return features

    # ======================================================
    # Extract Features
    # ======================================================

    def extract(
        self,
        X,
        sensor_columns,
        split_name="dataset"
    ):

        print("\n" + "=" * 70)
        print(
            f"WAVELET FEATURE EXTRACTION - "
            f"{split_name.upper()}"
        )
        print("=" * 70)

        print(
            f"Wavelet   : {self.wavelet}"
        )

        print(
            f"Level     : {self.level}"
        )

        print(
            f"Windows   : {X.shape[0]}"
        )

        print(
            f"Timesteps : {X.shape[1]}"
        )

        print(
            f"Sensors   : {len(sensor_columns)}"
        )

        feature_rows = []

        total_windows = len(X)

        # --------------------------------------------------
        # Process every window independently
        # --------------------------------------------------

        for window_index, window in enumerate(X):

            row = {
                "window_id": window_index
            }

            for sensor_index, sensor in enumerate(
                sensor_columns
            ):

                signal = window[
                    :,
                    sensor_index
                ]

                sensor_features = (
                    self._extract_sensor_features(
                        signal,
                        sensor
                    )
                )

                row.update(
                    sensor_features
                )

            feature_rows.append(row)

            if (
                window_index % 1000 == 0
                or window_index
                == total_windows - 1
            ):

                print(
                    f"Processed "
                    f"{window_index + 1}/"
                    f"{total_windows} windows..."
                )

        feature_df = pd.DataFrame(
            feature_rows
        )

        # ==================================================
        # Save Split-Specific Feature Matrix
        # ==================================================

        TABLES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        output_file = (
            TABLES_DIR
            / f"wavelet_features_{split_name}.csv"
        )

        feature_df.to_csv(
            output_file,
            index=False
        )

        # ==================================================
        # Feature Summary
        # ==================================================

        print("\n" + "-" * 70)

        print(
            f"Feature Matrix Shape : "
            f"{feature_df.shape}"
        )

        print(
            f"Total Features       : "
            f"{feature_df.shape[1] - 1}"
        )

        print("-" * 70)

        # ==================================================
        # Visualizations
        # ==================================================

        self._create_visualizations(
            feature_df,
            sensor_columns,
            split_name
        )

        print(
            "\n✓ Wavelet Feature Extraction Completed"
        )

        print(
            f"✓ Feature matrix saved:"
        )

        print(
            f"  {output_file}"
        )

        return feature_df

    # ======================================================
    # Visualizations
    # ======================================================

    def _create_visualizations(
        self,
        feature_df,
        sensor_columns,
        split_name
    ):

        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # --------------------------------------------------
        # Wavelet Energy
        # --------------------------------------------------

        energy_columns = [
            column
            for column in feature_df.columns
            if column.endswith(
                "_wavelet_energy"
            )
        ]

        if energy_columns:

            energy_values = (
                feature_df[
                    energy_columns
                ]
                .mean()
                .sort_values(
                    ascending=False
                )
            )

            plt.figure(
                figsize=(12, 6)
            )

            energy_values.plot(
                kind="bar"
            )

            plt.title(
                f"Mean Wavelet Energy "
                f"by Sensor - "
                f"{split_name.capitalize()}"
            )

            plt.xlabel("Sensor")

            plt.ylabel(
                "Mean Wavelet Energy"
            )

            plt.xticks(
                rotation=45,
                ha="right"
            )

            plt.grid(
                axis="y",
                alpha=0.3
            )

            plt.tight_layout()

            plt.savefig(
                FIGURES_DIR
                / f"wavelet_energy_"
                  f"{split_name}.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

        # --------------------------------------------------
        # Wavelet Entropy
        # --------------------------------------------------

        entropy_columns = [
            column
            for column in feature_df.columns
            if column.endswith(
                "_wavelet_entropy"
            )
        ]

        if entropy_columns:

            entropy_values = (
                feature_df[
                    entropy_columns
                ]
                .mean()
                .sort_values(
                    ascending=False
                )
            )

            plt.figure(
                figsize=(12, 6)
            )

            entropy_values.plot(
                kind="bar"
            )

            plt.title(
                f"Mean Wavelet Entropy "
                f"by Sensor - "
                f"{split_name.capitalize()}"
            )

            plt.xlabel("Sensor")

            plt.ylabel(
                "Mean Wavelet Entropy"
            )

            plt.xticks(
                rotation=45,
                ha="right"
            )

            plt.grid(
                axis="y",
                alpha=0.3
            )

            plt.tight_layout()

            plt.savefig(
                FIGURES_DIR
                / f"wavelet_entropy_"
                  f"{split_name}.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

        # --------------------------------------------------
        # Representative Decomposition
        # --------------------------------------------------

        if (
            self._last_X is not None
            and len(sensor_columns) > 0
            and len(self._last_X) > 0
        ):

            sensor_index = 0

            signal = self._last_X[
                0,
                :,
                sensor_index
            ]

            wavelet = pywt.Wavelet(
                self.wavelet
            )

            maximum_level = (
                pywt.dwt_max_level(
                    data_len=len(signal),
                    filter_len=wavelet.dec_len
                )
            )

            level = min(
                self.level,
                maximum_level
            )

            coefficients = pywt.wavedec(
                signal,
                self.wavelet,
                level=level,
                mode="symmetric"
            )

            plt.figure(
                figsize=(12, 7)
            )

            for index, coefficient in enumerate(
                coefficients
            ):

                plt.plot(
                    coefficient,
                    label=f"Coefficient {index}"
                )

            plt.title(
                f"Wavelet Decomposition - "
                f"{sensor_columns[0]} - "
                f"{split_name.capitalize()}"
            )

            plt.xlabel(
                "Coefficient Index"
            )

            plt.ylabel(
                "Amplitude"
            )

            plt.legend()

            plt.grid(
                alpha=0.3
            )

            plt.tight_layout()

            plt.savefig(
                FIGURES_DIR
                / f"wavelet_coefficients_"
                  f"{split_name}.png",
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

    # ======================================================
    # Store Input
    # ======================================================

    def set_input(self, X):

        self._last_X = X