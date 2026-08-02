"""
Feature Normalization Module

Uses StandardScaler and supports:
1. fit()
2. transform()
3. fit_transform()
4. save()
5. load()
"""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


class DataNormalizer:

    def __init__(self):

        self.scaler = StandardScaler()

        self.exclude_columns = [
            "engine_id",
            "cycle",
            "RUL"
        ]

    def fit(self, df: pd.DataFrame):

        feature_columns = [
            col for col in df.columns
            if col not in self.exclude_columns
        ]

        self.scaler.fit(df[feature_columns])

        self.feature_columns = feature_columns

        return self

    def transform(self, df: pd.DataFrame):

        df = df.copy()

        df[self.feature_columns] = self.scaler.transform(
            df[self.feature_columns]
        )

        return df

    def fit_transform(self, df: pd.DataFrame):

        self.fit(df)

        return self.transform(df)

    def save(self, filepath):

        Path(filepath).parent.mkdir(
            parents=True,
            exist_ok=True
        )

        joblib.dump(self.scaler, filepath)

        print(f"\n✓ Scaler saved to {filepath}")

    def load(self, filepath):

        self.scaler = joblib.load(filepath)

        print(f"\n✓ Scaler loaded from {filepath}")