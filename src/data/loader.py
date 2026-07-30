"""
NASA C-MAPSS Dataset Loader
"""

import pandas as pd
from pathlib import Path


class CMAPSSLoader:
    """Loads NASA C-MAPSS datasets."""

    def __init__(self, train_path: Path, test_path: Path, rul_path: Path):
        self.train_path = train_path
        self.test_path = test_path
        self.rul_path = rul_path

        self.columns = (
            ["engine_id", "cycle"]
            + [f"setting_{i}" for i in range(1, 4)]
            + [f"sensor_{i}" for i in range(1, 22)]
        )

    def _load_file(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            engine="python"
        )

        # Remove empty columns present in NASA files
        df = df.iloc[:, :26]
        df.columns = self.columns

        return df

    def load_train(self) -> pd.DataFrame:
        return self._load_file(self.train_path)

    def load_test(self) -> pd.DataFrame:
        return self._load_file(self.test_path)

    def load_rul(self) -> pd.DataFrame:
        return pd.read_csv(self.rul_path, header=None, names=["RUL"])