"""
Engine-wise Train / Validation / Test Splitter

Splits C-MAPSS data by engine_id to prevent temporal
and engine-level data leakage.
"""

import numpy as np
import pandas as pd


class EngineWiseSplitter:

    def __init__(
        self,
        train_ratio=0.70,
        validation_ratio=0.15,
        test_ratio=0.15,
        random_state=42
    ):

        if not np.isclose(
            train_ratio + validation_ratio + test_ratio,
            1.0
        ):
            raise ValueError(
                "Train, validation and test ratios must sum to 1.0."
            )

        self.train_ratio = train_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

    def split(self, df):

        engines = np.array(
            sorted(df["engine_id"].unique())
        )

        rng = np.random.default_rng(
            self.random_state
        )

        rng.shuffle(engines)

        total_engines = len(engines)

        train_count = int(
            total_engines * self.train_ratio
        )

        validation_count = int(
            total_engines * self.validation_ratio
        )

        train_engines = engines[
            :train_count
        ]

        validation_engines = engines[
            train_count:
            train_count + validation_count
        ]

        test_engines = engines[
            train_count + validation_count:
        ]

        train_df = df[
            df["engine_id"].isin(train_engines)
        ].copy()

        validation_df = df[
            df["engine_id"].isin(validation_engines)
        ].copy()

        test_df = df[
            df["engine_id"].isin(test_engines)
        ].copy()

        train_df.sort_values(
            ["engine_id", "cycle"],
            inplace=True
        )

        validation_df.sort_values(
            ["engine_id", "cycle"],
            inplace=True
        )

        test_df.sort_values(
            ["engine_id", "cycle"],
            inplace=True
        )

        train_df.reset_index(
            drop=True,
            inplace=True
        )

        validation_df.reset_index(
            drop=True,
            inplace=True
        )

        test_df.reset_index(
            drop=True,
            inplace=True
        )

        print("\n" + "=" * 70)
        print("ENGINE-WISE DATA SPLIT")
        print("=" * 70)

        print(
            f"Total Engines      : {total_engines}"
        )

        print(
            f"Training Engines   : "
            f"{len(train_engines)}"
        )

        print(
            f"Validation Engines : "
            f"{len(validation_engines)}"
        )

        print(
            f"Test Engines       : "
            f"{len(test_engines)}"
        )

        print("\nTraining Engine IDs:")
        print(
            sorted(train_engines.tolist())
        )

        print("\nValidation Engine IDs:")
        print(
            sorted(validation_engines.tolist())
        )

        print("\nTest Engine IDs:")
        print(
            sorted(test_engines.tolist())
        )

        print("\nDataset Shapes:")

        print(
            f"Train      : {train_df.shape}"
        )

        print(
            f"Validation : {validation_df.shape}"
        )

        print(
            f"Test       : {test_df.shape}"
        )

        # Safety check
        train_set = set(train_engines)
        validation_set = set(validation_engines)
        test_set = set(test_engines)

        assert train_set.isdisjoint(validation_set)
        assert train_set.isdisjoint(test_set)
        assert validation_set.isdisjoint(test_set)

        print(
            "\n✓ No engine appears in more than one split."
        )

        return (
            train_df,
            validation_df,
            test_df
        )