"""
EDA Visualizations
"""

import matplotlib.pyplot as plt

from configs.config import FIGURES_DIR, FIG_DPI


class EDAVisualizer:

    def __init__(self, df):
        self.df = df

    def engine_life_distribution(self):

        engine_cycles = (
            self.df.groupby("engine_id")["cycle"]
            .max()
            .sort_values()
        )

        plt.figure(figsize=(10, 6))

        plt.hist(engine_cycles, bins=20)

        plt.title("Engine Life Distribution")
        plt.xlabel("Maximum Cycles")
        plt.ylabel("Number of Engines")

        plt.grid(alpha=0.3)

        plt.tight_layout()

        plt.savefig(
            FIGURES_DIR / "engine_life_distribution.png",
            dpi=FIG_DPI
        )

        plt.show()