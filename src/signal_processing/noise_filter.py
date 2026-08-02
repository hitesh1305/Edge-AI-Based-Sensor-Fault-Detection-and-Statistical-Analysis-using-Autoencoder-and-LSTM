"""
Noise Filtering using Savitzky-Golay Filter

Applies smoothing independently to each engine
to preserve degradation trajectories.
"""

import pandas as pd
from scipy.signal import savgol_filter


class NoiseFilter:

    def __init__(
        self,
        window_length=11,
        polyorder=2
    ):

        self.window_length = window_length
        self.polyorder = polyorder

    def apply(self, df: pd.DataFrame):

        filtered = df.copy()

        sensor_columns = [
            col
            for col in filtered.columns
            if col.startswith("sensor_")
        ]

        print("\n" + "=" * 70)
        print("NOISE FILTERING")
        print("=" * 70)

        print(f"Total Engines : {filtered['engine_id'].nunique()}")
        print(f"Sensors       : {len(sensor_columns)}")

        # ----------------------------------------
        # Filter every engine independently
        # ----------------------------------------

        for engine in filtered["engine_id"].unique():

            engine_mask = filtered["engine_id"] == engine

            engine_data = filtered.loc[engine_mask]

            for sensor in sensor_columns:

                signal = engine_data[sensor].values

                # Skip if sequence is too short
                if len(signal) < self.window_length:
                    continue

                filtered_signal = savgol_filter(
                    signal,
                    window_length=self.window_length,
                    polyorder=self.polyorder,
                    mode="nearest"
                )

                filtered.loc[
                    engine_mask,
                    sensor
                ] = filtered_signal

        print("\n✓ Noise filtering completed.")

        return filtered