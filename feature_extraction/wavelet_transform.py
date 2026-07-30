import os
import numpy as np
import pandas as pd
import pywt
import matplotlib

# Use non-GUI backend
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def apply_wavelet_transform(df):
    """
    Applies Discrete Wavelet Transform (DWT) on the vibration signal.

    Parameters
    ----------
    df : pandas.DataFrame
        Filtered sensor dataframe.

    Returns
    -------
    dict
        {
            "features": DataFrame,
            "coefficients": coeffs
        }
    """

    print("\nApplying Wavelet Transform...")

    # --------------------------------------------------
    # Create Output Folders
    # --------------------------------------------------
    os.makedirs("output", exist_ok=True)
    os.makedirs("output/plots", exist_ok=True)

    # --------------------------------------------------
    # Select Vibration Signal
    # --------------------------------------------------
    signal = df["vibration"].to_numpy(copy=True)

    # --------------------------------------------------
    # Apply DWT
    # --------------------------------------------------
    wavelet = "db4"
    level = 3

    coeffs = pywt.wavedec(
        signal,
        wavelet=wavelet,
        level=level
    )

    # coeffs = [A3, D3, D2, D1]
    labels = ["A3", "D3", "D2", "D1"]

    # --------------------------------------------------
    # Extract Features
    # --------------------------------------------------
    features = []

    for label, coeff in zip(labels, coeffs):

        mean = np.mean(coeff)
        std = np.std(coeff)
        energy = np.sum(coeff ** 2)

        features.append({
            "Coefficient": label,
            "Mean": mean,
            "Std Dev": std,
            "Energy": energy
        })

    feature_df = pd.DataFrame(features)

    # --------------------------------------------------
    # Save Features
    # --------------------------------------------------
    feature_df.to_csv(
        "output/wavelet_features.csv",
        index=False
    )

    # --------------------------------------------------
    # Plot Wavelet Decomposition
    # --------------------------------------------------
    plt.figure(figsize=(12, 8))

    plt.subplot(5, 1, 1)
    plt.plot(signal)
    plt.title("Original Vibration Signal")

    for i, coeff in enumerate(coeffs):

        plt.subplot(5, 1, i + 2)
        plt.plot(coeff)
        plt.title(labels[i])

    plt.tight_layout()

    plt.savefig(
        "output/plots/wavelet_decomposition.png"
    )

    plt.close()

    # --------------------------------------------------
    # Print Summary
    # --------------------------------------------------
    print("\nWavelet Features")
    print(feature_df)

    print("\nWavelet decomposition plot saved.")

    return {
        "features": feature_df,
        "coefficients": coeffs
    }