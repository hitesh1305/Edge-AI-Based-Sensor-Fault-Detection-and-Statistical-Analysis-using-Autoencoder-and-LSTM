"""
Sliding Window Generator
"""

import numpy as np
import pandas as pd


class WindowGenerator:
    """
    Generates sliding windows for sequence models.
    """

    def __init__(self, window_size=30):
        self.window_size = window_size

    def generate(self, df: pd.DataFrame):

        feature_cols = [
            col for col in df.columns
            if col not in ["engine_id", "cycle", "RUL"]
        ]

        X = []
        y = []

        # Process each engine independently
        for _, engine_data in df.groupby("engine_id"):

            engine_data = engine_data.reset_index(drop=True)

            features = engine_data[feature_cols].values
            rul = engine_data["RUL"].values

            for i in range(len(engine_data) - self.window_size + 1):

                X.append(features[i:i + self.window_size])
                y.append(rul[i + self.window_size - 1])

        X = np.array(X)
        y = np.array(y)

        print("\n========== WINDOW GENERATION ==========")
        print(f"Window Size : {self.window_size}")
        print(f"Samples     : {len(X)}")
        print(f"X Shape     : {X.shape}")
        print(f"y Shape     : {y.shape}")

        return X, y