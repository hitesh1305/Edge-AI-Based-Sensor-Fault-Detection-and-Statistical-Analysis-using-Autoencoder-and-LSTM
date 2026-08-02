"""
Data Preprocessing Module
"""

import pandas as pd


class Preprocessor:
    """
    Handles preprocessing of the NASA C-MAPSS dataset.
    """

    def __init__(self):
        self.columns_to_remove = [
            "setting_3",
            "sensor_1",
            "sensor_5",
            "sensor_10",
            "sensor_16",
            "sensor_18",
            "sensor_19",
        ]

    def remove_constant_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove constant features from the dataset.
        """

        cleaned_df = df.drop(columns=self.columns_to_remove)

        print("\n========== PREPROCESSING ==========")
        print(f"Original Shape : {df.shape}")
        print(f"Processed Shape: {cleaned_df.shape}")

        print("\nRemoved Features:")
        for feature in self.columns_to_remove:
            print(f"  ✓ {feature}")

        return cleaned_df

    def add_rul(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Remaining Useful Life (RUL) for every row.
        RUL = Max Cycle of Engine - Current Cycle
        """

        df = df.copy()

        max_cycles = (
            df.groupby("engine_id")["cycle"]
            .max()
            .rename("max_cycle")
        )

        df = df.merge(
            max_cycles,
            on="engine_id",
            how="left"
        )

        df["RUL"] = df["max_cycle"] - df["cycle"]

        df.drop(columns=["max_cycle"], inplace=True)

        print("\n========== RUL GENERATION ==========")
        print(f"Minimum RUL : {df['RUL'].min()}")
        print(f"Maximum RUL : {df['RUL'].max()}")
        print(f"Average RUL : {df['RUL'].mean():.2f}")

        print("\nFirst 10 Samples:\n")
        print(
            df[
                ["engine_id", "cycle", "RUL"]
            ].head(10)
        )

        return df