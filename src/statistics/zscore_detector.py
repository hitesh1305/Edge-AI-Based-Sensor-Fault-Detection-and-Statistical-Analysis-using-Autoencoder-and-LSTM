"""
Statistical Anomaly Detection using Z-Score

This module computes sensor-wise Z-scores,
creates a statistical anomaly report,
and adds only two useful features:
    • max_z_score
    • zscore_flag
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class ZScoreDetector:

    def __init__(self, threshold=3.0):

        self.threshold = threshold

    def detect(self, df: pd.DataFrame):

        df = df.copy()

        sensor_columns = [
            c for c in df.columns
            if c.startswith("sensor_")
        ]

        print("\n" + "=" * 70)
        print("Z-SCORE STATISTICAL ANALYSIS")
        print("=" * 70)

        report = []

        max_z = np.zeros(len(df))

        anomaly_flag = np.zeros(len(df), dtype=int)

        heatmap_matrix = []

        # --------------------------------------------------
        # Process each sensor independently
        # --------------------------------------------------

        for sensor in sensor_columns:

            mean = df[sensor].mean()

            std = df[sensor].std()

            if std == 0:
                z = np.zeros(len(df))
            else:
                z = np.abs((df[sensor] - mean) / std)

            heatmap_matrix.append(z)

            max_z = np.maximum(max_z, z)

            anomaly_flag = np.where(
                z > self.threshold,
                1,
                anomaly_flag
            )

            anomalies = int((z > self.threshold).sum())

            report.append({
                "Sensor": sensor,
                "Mean": round(mean, 4),
                "Std": round(std, 4),
                "Max Z-Score": round(z.max(), 4),
                "Anomalies": anomalies,
                "Percentage": round(
                    anomalies / len(df) * 100,
                    3
                )
            })

            print(
                f"{sensor:<12}"
                f"  Max Z = {z.max():6.2f}"
                f"   Anomalies = {anomalies}"
            )

        # --------------------------------------------------
        # Keep ONLY useful outputs
        # --------------------------------------------------

        df["max_z_score"] = max_z

        df["zscore_flag"] = anomaly_flag

        report_df = pd.DataFrame(report)

        # --------------------------------------------------
        # Console Summary
        # --------------------------------------------------

        print("\n" + "-" * 70)

        print(
            f"Total Statistical Anomalies : "
            f"{df['zscore_flag'].sum()}"
        )

        print(
            f"Overall Percentage          : "
            f"{df['zscore_flag'].mean()*100:.2f}%"
        )

        print(
            f"Maximum Z Score             : "
            f"{df['max_z_score'].max():.2f}"
        )

        print("-" * 70)

        # --------------------------------------------------
        # Save Report
        # --------------------------------------------------

        tables = Path("output/tables")
        tables.mkdir(parents=True, exist_ok=True)

        report_df.to_csv(
            tables / "zscore_report.csv",
            index=False
        )

        # --------------------------------------------------
        # Histogram
        # --------------------------------------------------

        figures = Path("output/figures")
        figures.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(8,5))

        plt.hist(
            df["max_z_score"],
            bins=50,
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
            figures / "zscore_distribution.png",
            dpi=300
        )

        plt.close()

        # --------------------------------------------------
        # Heatmap
        # --------------------------------------------------

        plt.figure(figsize=(14,7))

        sns.heatmap(
            np.array(heatmap_matrix)[:, :300],
            cmap="viridis",
            cbar=True
        )

        plt.title(
            "Sensor-wise Z-Score Heatmap"
        )

        plt.xlabel("Samples")

        plt.ylabel("Sensors")

        plt.tight_layout()

        plt.savefig(
            figures / "zscore_heatmap.png",
            dpi=300
        )

        plt.close()

        print("\n✓ Z-Score Report Saved")

        print("✓ Distribution Plot Saved")

        print("✓ Heatmap Saved")

        return df