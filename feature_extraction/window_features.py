import os
import numpy as np
import pandas as pd
import pywt

from scipy.stats import skew, kurtosis


def extract_window_features(X):
    """
    Extract statistical and wavelet features from each sliding window.

    Parameters
    ----------
    X : numpy.ndarray
        Shape:
        (num_windows, window_size, num_features)

    Returns
    -------
    pandas.DataFrame
    """

    print("\nExtracting Advanced Window Features...")

    feature_names = [
        "temperature",
        "vibration",
        "humidity",
        "pressure",
        "energy_consumption"
    ]

    extracted_features = []

    for window in X:

        row = {}

        # ==========================================
        # Statistical Features
        # ==========================================

        for i, feature in enumerate(feature_names):

            signal = window[:, i]

            row[f"{feature}_mean"] = np.mean(signal)
            row[f"{feature}_std"] = np.std(signal)
            row[f"{feature}_min"] = np.min(signal)
            row[f"{feature}_max"] = np.max(signal)

            row[f"{feature}_variance"] = np.var(signal)

            row[f"{feature}_rms"] = np.sqrt(np.mean(signal**2))

            row[f"{feature}_ptp"] = np.ptp(signal)

            row[f"{feature}_skewness"] = skew(signal)

            row[f"{feature}_kurtosis"] = kurtosis(signal)

        # ==========================================
        # Wavelet Features (Vibration Only)
        # ==========================================

        vibration_signal = window[:, 1]

        coeffs = pywt.wavedec(
            vibration_signal,
            wavelet="db4",
            level=3
        )

        labels = ["A3", "D3", "D2", "D1"]

        for label, coeff in zip(labels, coeffs):

            row[f"{label}_energy"] = np.sum(coeff**2)

        extracted_features.append(row)

    feature_df = pd.DataFrame(extracted_features)

    os.makedirs("output", exist_ok=True)

    feature_df.to_csv(
        "output/window_features.csv",
        index=False
    )

    print("\n✓ Advanced Window Feature Extraction Completed")

    print(f"Feature Matrix Shape : {feature_df.shape}")

    return feature_df