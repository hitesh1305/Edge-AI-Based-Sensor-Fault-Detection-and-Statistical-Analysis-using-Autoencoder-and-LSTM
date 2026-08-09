"""
EDA Visualizations
"""

import matplotlib.pyplot as plt
import seaborn as sns

from configs.config import FIGURES_DIR, FIG_DPI


class EDAVisualizer:
    """
    Generates publication-quality EDA visualizations for the NASA C-MAPSS dataset.
    """

    def __init__(self, df):
        self.df = df

    def engine_life_distribution(self):
        """
        Plot the distribution of engine life (maximum cycles).
        """

        engine_cycles = (
            self.df.groupby("engine_id")["cycle"]
            .max()
            .sort_values()
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(
            engine_cycles,
            bins=20,
            color="steelblue",
            edgecolor="black"
        )

        ax.set_title("Engine Life Distribution")
        ax.set_xlabel("Maximum Cycles")
        ax.set_ylabel("Number of Engines")

        ax.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / "engine_life_distribution.png",
            dpi=FIG_DPI,
            bbox_inches="tight"
        )

        plt.show()

    def sensor_variance(self):
        """
        Plot sensor variance ranking using a logarithmic scale.
        """

        sensor_cols = [
            col for col in self.df.columns
            if col.startswith("sensor_")
        ]

        variance = (
            self.df[sensor_cols]
            .var()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(12, 6))

        variance.plot(
            kind="bar",
            ax=ax,
            color="steelblue"
        )

        ax.set_yscale("log")

        ax.set_title("Sensor Variance Ranking")
        ax.set_xlabel("Sensors")
        ax.set_ylabel("Variance (Log Scale)")

        ax.grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / "sensor_variance.png",
            dpi=FIG_DPI,
            bbox_inches="tight"
        )

        plt.show()

        print("\n========== SENSOR VARIANCE ==========\n")
        print(variance)

    def correlation_heatmap(self):
        """
        Plot correlation heatmap for all sensors.
        """

        sensor_cols = [
            col for col in self.df.columns
            if col.startswith("sensor_")
        ]

        corr = self.df[sensor_cols].corr()

        fig, ax = plt.subplots(figsize=(14, 10))

        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=0.5,
            ax=ax
        )

        ax.set_title("Sensor Correlation Heatmap")

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / "correlation_heatmap.png",
            dpi=FIG_DPI,
            bbox_inches="tight"
        )

        plt.close()