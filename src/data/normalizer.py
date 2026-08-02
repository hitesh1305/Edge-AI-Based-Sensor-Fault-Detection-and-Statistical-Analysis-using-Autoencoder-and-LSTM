"""
Data Normalization Module
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataNormalizer:
    """
    Normalizes sensor and operational setting features.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, df: pd.DataFrame):

        df = df.copy()

        feature_cols = [
            col for col in df.columns
            if col not in ["engine_id", "cycle", "RUL"]
        ]

        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])

        print("\n========== NORMALIZATION ==========")
        print(f"Scaled Features : {len(feature_cols)}")
        print(feature_cols)

        return df

    def get_scaler(self):
        return self.scaler