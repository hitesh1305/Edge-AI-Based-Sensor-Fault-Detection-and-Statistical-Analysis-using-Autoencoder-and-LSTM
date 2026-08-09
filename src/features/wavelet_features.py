"""
Wavelet-Based Feature Extraction

Extracts time-domain and wavelet-domain features
from each sliding window.

The extraction is performed independently for every
window to avoid temporal information leakage.
"""

from pathlib import Path

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
    """
    Extracts fixed-length statistical and wavelet features
    from sliding-window sensor sequences.
    """

    def __init__(
        self,
        wavelet=WAVELET_NAME,
        level=WAVELET_LEVEL
    ):
        self.wavelet = wavelet
        self.level = level

    # ======================================================
    # Basic Statistical Features
    # ======================================================

    @staticmethod
    def _statistical_features(signal):
        """
        Calculate time-domain statistical features.
        """

        signal = np.asarray(signal, dtype=np.float64)

        mean = np.mean(signal)

        std = np.std(signal)

        variance = np.var(signal)

        rms = np.sqrt(np.mean(signal ** 2))

        maximum = np.max(signal)

        minimum = np.min(signal)

        peak_to_peak = maximum - minimum

        return {
            "mean": mean,
            "std": std,
            "variance": variance,
            "rms": rms,
            "max": maximum,
            "min": minimum,
            "peak_to_peak": peak_to_peak,
        }

    # ======================================================
    # Wavelet Energy
    # ======================================================

    @staticmethod
    def _wavelet_energy(coefficients):
        """
        Calculate total wavelet energy.
        """

        energy = 0.0

        for coefficient in coefficients:
            energy += np.sum(np.square(coefficient))

        return energy

    # ======================================================
    # Wavelet Entropy
    # ======================================================

    @staticmethod
    def _wavelet_entropy(coefficients):
        """
        Calculate normalized Shannon entropy of wavelet
        coefficient energy distribution.
        """

        energies = []

        for coefficient in coefficients:

            energy = np.square(coefficient)

            energies.extend(energy.tolist())

        energies = np.asarray(energies, dtype=np.float64)

        total_energy = np.sum(energies)

        if total_energy <= 0:
            return 0.0

        probabilities = energies / total_energy

        probabilities = probabilities[
            probabilities > 0
        ]

        entropy = -np.sum(
            probabilities * np.log2(probabilities)
        )

        return entropy

    # ======================================================
    # Single Sensor Feature Extraction
    # ======================================================

    def _extract_sensor_features(
        self,
        signal,
        prefix
    ):
        """
        Extract features from one sensor within one window.
        """

        features = {}

        # --------------------------------------------------
        # Time-domain features
        # --------------------------------------------------

        statistics = self._statistical_features(signal)

        for name, value in statistics.items():

            features[
                f"{prefix}_{name}"
            ] = value

        # --------------------------------------------------
        # Wavelet decomposition
        # --------------------------------------------------

        wavelet = pywt.Wavelet(self.wavelet)

        maximum_level = pywt.dwt_max_level(
            data_len=len(signal),
            filter_len=wavelet.dec_len
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
        # Wavelet features
        # --------------------------------------------------

        wavelet_energy = self._wavelet_energy(
            coefficients
        )

        wavelet_entropy = self._wavelet_entropy(
            coefficients
        )

        features[
            f"{prefix}_wavelet_energy"
        ] = wavelet_energy

        features[
            f"{prefix}_wavelet_entropy"
        ] = wavelet_entropy

        return features

    # ======================================================
    # Extract Features from All Windows
    # ======================================================

    def extract(
        self,
        X,
        sensor_columns
    ):
        """
        Parameters
        ----------
        X :
            Sliding-window array with shape:

            (samples, timesteps, features)

        sensor_columns :
            Names of sensor features contained in X.

        Returns
        -------
        feature_df :
            Fixed-length feature matrix.
        """

        print("\n" + "=" * 70)
        print("WAVELET FEATURE EXTRACTION")
        print("=" * 70)

        print(f"Wavelet : {self.wavelet}")
        print(f"Level   : {self.level}")

        print(f"Windows : {X.shape[0]}")
        print(f"Timesteps : {X.shape[1]}")
        print(f"Sensors : {len(sensor_columns)}")

        feature_rows = []

        total_windows = len(X)

        for window_index, window in enumerate(X):

            row = {
                "window_id": window_index
            }

            for sensor_index, sensor in enumerate(
                sensor_columns
            ):

                signal = window[:, sensor_index]

                prefix = sensor

                sensor_features = (
                    self._extract_sensor_features(
                        signal,
                        prefix
                    )
                )

                row.update(sensor_features)

            feature_rows.append(row)

            # Progress indicator
            if (
                window_index % 1000 == 0
                or window_index == total_windows - 1
            ):

                print(
                    f"Processed "
                    f"{window_index + 1}/"
                    f"{total_windows} windows..."
                )

        feature_df = pd.DataFrame(feature_rows)

        # ==================================================
        # Save Feature Matrix
        # ==================================================

        TABLES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        feature_df.to_csv(
            TABLES_DIR / "wavelet_features.csv",
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
        # Generate Visualizations
        # ==================================================

        self._create_visualizations(
            feature_df,
            sensor_columns
        )

        print("\n✓ Wavelet Feature Extraction Completed")

        print(
            f"✓ Feature matrix saved to "
            f"{TABLES_DIR / 'wavelet_features.csv'}"
        )

        return feature_df

    # ======================================================
    # Visualizations
    # ======================================================

    def _create_visualizations(
        self,
        feature_df,
        sensor_columns
    ):
        """
        Generate representative wavelet feature plots.
        """

        FIGURES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        # --------------------------------------------------
        # Wavelet Energy Distribution
        # --------------------------------------------------

        energy_columns = [
            column
            for column in feature_df.columns
            if column.endswith("_wavelet_energy")
        ]

        if energy_columns:

            energy_values = (
                feature_df[energy_columns]
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
                "Mean Wavelet Energy by Sensor"
            )

            plt.xlabel("Sensor")

            plt.ylabel("Mean Wavelet Energy")

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
                FIGURES_DIR /
                "wavelet_energy_distribution.png",
                dpi=300
            )

            plt.close()

        # --------------------------------------------------
        # Wavelet Entropy Distribution
        # --------------------------------------------------

        entropy_columns = [
            column
            for column in feature_df.columns
            if column.endswith("_wavelet_entropy")
        ]

        if entropy_columns:

            entropy_values = (
                feature_df[entropy_columns]
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
                "Mean Wavelet Entropy by Sensor"
            )

            plt.xlabel("Sensor")

            plt.ylabel("Mean Wavelet Entropy")

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
                FIGURES_DIR /
                "wavelet_entropy_distribution.png",
                dpi=300
            )

            plt.close()

        # --------------------------------------------------
        # Representative Sensor Wavelet Decomposition
        # --------------------------------------------------

        if len(sensor_columns) > 0:

            sensor_index = 0

            signal = (
                np.asarray(
                    self._last_X[
                        0,
                        :,
                        sensor_index
                    ]
                )
            )

            coefficients = pywt.wavedec(
                signal,
                self.wavelet,
                level=min(
                    self.level,
                    pywt.dwt_max_level(
                        len(signal),
                        pywt.Wavelet(
                            self.wavelet
                        ).dec_len
                    )
                ),
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
                f"{sensor_columns[0]}"
            )

            plt.xlabel("Coefficient Index")

            plt.ylabel("Amplitude")

            plt.legend()

            plt.grid(
                alpha=0.3
            )

            plt.tight_layout()

            plt.savefig(
                FIGURES_DIR /
                "wavelet_coefficients.png",
                dpi=300
            )

            plt.close()

    # ======================================================
    # Internal Storage
    # ======================================================

    def set_input(self, X):
        """
        Store input windows for visualization.
        """

        self._last_X = X