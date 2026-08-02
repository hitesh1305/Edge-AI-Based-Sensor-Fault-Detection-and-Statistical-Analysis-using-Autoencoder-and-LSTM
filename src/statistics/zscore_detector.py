"""
Statistical Anomaly Detection using Z-Score
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ZScoreDetector:
    """
    Performs statistical anomaly detection using
    sensor-wise Z-Score analysis.
    """

    def __init__(self, threshold=3.0):

        self.threshold = threshold

    def detect(self, df: pd.DataFrame):

        df = df.copy()

        sensor_columns = [
            col
            for col in df.columns
            if col.startswith("sensor_")
        ]

        print("\n" + "=" * 70)
        print("Z-SCORE STATISTICAL ANALYSIS")
        print("=" * 70)

        report = []

        max_scores = np.zeros(len(df))

        flags = np.zeros(len(df), dtype=int)

        for sensor in sensor_columns:

            mean = df[sensor].mean()

            std = df[sensor].std()

            if std == 0:
                z = np.zeros(len(df))
            else:
                z = np.abs((df[sensor] - mean) / std)

            df[f"{sensor}_z"] = z

            anomalies = z > self.threshold

            anomaly_count = int(anomalies.sum())

            anomaly_percent = anomaly_count / len(df) * 100

            report.append({
                "Sensor": sensor,
                "Mean": mean,
                "Std": std,
                "Anomalies": anomaly_count,
                "Percentage": anomaly_percent
            })

            max_scores = np.maximum(max_scores, z)

            flags = np.where(anomalies, 1, flags)

            print(
                f"{sensor:<12}"
                f" | "
                f"Anomalies: {anomaly_count:<5}"
                f" ({anomaly_percent:.2f}%)"
            )

        df["max_z_score"] = max_scores

        df["zscore_flag"] = flags

        report_df = pd.DataFrame(report)

        print("\n" + "-" * 70)

        print(
            f"Total Statistical Anomalies : "
            f"{df['zscore_flag'].sum()}"
        )

        print(
            f"Overall Percentage          : "
            f"{df['zscore_flag'].mean()*100:.2f}%"
        )

        print("-" * 70)

        # -----------------------------------------------------
        # Save CSV
        # -----------------------------------------------------

        output_table = Path("output/tables")

        output_table.mkdir(
            parents=True,
            exist_ok=True
        )

        report_df.to_csv(
            output_table / "zscore_report.csv",
            index=False
        )

        # -----------------------------------------------------
        # Histogram
        # -----------------------------------------------------

        output_fig = Path("output/figures")

        output_fig.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.figure(figsize=(8,5))

        plt.hist(
            df["max_z_score"],
            bins=60,
            edgecolor="black"
        )

        plt.axvline(
            self.threshold,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Threshold"
        )

        plt.title("Maximum Z-Score Distribution")

        plt.xlabel("Maximum Z-Score")

        plt.ylabel("Frequency")

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            output_fig / "zscore_distribution.png",
            dpi=300
        )

        plt.close()

        # -----------------------------------------------------
        # Heatmap
        # -----------------------------------------------------

        z_columns = [
            c
            for c in df.columns
            if c.endswith("_z")
        ]

        plt.figure(figsize=(14,8))

        sns.heatmap(
            df[z_columns].head(300),
            cmap="viridis",
            cbar=True
        )

        plt.title(
            "Sensor-wise Z-Score Heatmap (First 300 Samples)"
        )

        plt.tight_layout()

        plt.savefig(
            output_fig / "zscore_heatmap.png",
            dpi=300
        )

        plt.close()

        print("\n✓ Z-Score report saved")

        print("✓ Distribution plot saved")

        print("✓ Heatmap saved")

        return df