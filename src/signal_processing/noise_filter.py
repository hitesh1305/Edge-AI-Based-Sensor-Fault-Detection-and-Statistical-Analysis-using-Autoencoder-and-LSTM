"""
Noise Filtering using Savitzky-Golay Filter
"""

import pandas as pd
from scipy.signal import savgol_filter


class NoiseFilter:
    """
    Applies Savitzky-Golay smoothing
    to all sensor signals.
    """

    def __init__(self,
                 window_length=11,
                 polyorder=2):

        self.window_length = window_length
        self.polyorder = polyorder

    def apply(self, df: pd.DataFrame):

        filtered = df.copy()

        sensor_columns = [
            col
            for col in filtered.columns
            if col.startswith("sensor_")
        ]

        print("\n========== NOISE FILTER ==========")

        print(f"Filtering {len(sensor_columns)} sensors...")

        for sensor in sensor_columns:

            filtered[sensor] = savgol_filter(
                filtered[sensor],
                window_length=self.window_length,
                polyorder=self.polyorder,
                mode="nearest"
            )

        print("✓ Noise filtering completed")

        return filtered