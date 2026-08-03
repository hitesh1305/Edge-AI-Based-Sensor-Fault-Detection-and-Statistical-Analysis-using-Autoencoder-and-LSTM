"""
Sliding Window Generator

Creates sequential windows for each engine independently.
Each window is assigned the RUL of its last time step.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from configs.config import (
    WINDOW_SIZE,
    TABLES_DIR
)


class WindowGenerator:

    def __init__(
        self,
        window_size=WINDOW_SIZE,
        stride=1
    ):

        self.window_size = window_size
        self.stride = stride

    def generate(self, df):

        print("\n" + "=" * 70)
        print("SLIDING WINDOW GENERATION")
        print("=" * 70)

        feature_columns = [
            col
            for col in df.columns
            if col not in [
                "engine_id",
                "cycle",
                "RUL",
                "max_z_score",
                "zscore_flag"
            ]
        ]

        X = []
        y = []

        metadata = []

        window_id = 0

        engines = sorted(df["engine_id"].unique())

        print(f"Total Engines : {len(engines)}")
        print(f"Window Size   : {self.window_size}")
        print(f"Stride        : {self.stride}")

        for engine in engines:

            engine_df = (
                df[df["engine_id"] == engine]
                .sort_values("cycle")
                .reset_index(drop=True)
            )

            total_cycles = len(engine_df)

            for start in range(
                0,
                total_cycles - self.window_size + 1,
                self.stride
            ):

                end = start + self.window_size

                window = engine_df.iloc[start:end]

                X.append(
                    window[feature_columns].values
                )

                y.append(
                    window["RUL"].iloc[-1]
                )

                metadata.append({
                    "window_id": window_id,
                    "engine_id": engine,
                    "start_cycle": int(window["cycle"].iloc[0]),
                    "end_cycle": int(window["cycle"].iloc[-1]),
                    "target_rul": int(window["RUL"].iloc[-1])
                })

                window_id += 1

        X = np.asarray(X, dtype=np.float32)

        y = np.asarray(y, dtype=np.float32)

        metadata = pd.DataFrame(metadata)

        TABLES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        metadata.to_csv(
            TABLES_DIR / "window_information.csv",
            index=False
        )

        print("\n✓ Window information saved")

        print("\nTotal Windows :", len(X))
        print("Feature Shape :", X.shape)
        print("Target Shape  :", y.shape)

        print("\nFirst Window Shape")
        print(X[0].shape)

        print("\nFirst Window Target (RUL)")
        print(y[0])

        print("\nFirst Feature Vector")
        print(X[0][0])

        return X, y, metadata