"""
Sliding Window Generator

Generates engine-wise sequential windows for the NASA C-MAPSS
dataset and stores metadata separately for train, validation,
and test splits.
"""

import numpy as np
import pandas as pd

from configs.config import WINDOW_SIZE, TABLES_DIR


class WindowGenerator:

    def __init__(
        self,
        window_size=WINDOW_SIZE,
        stride=1
    ):
        self.window_size = window_size
        self.stride = stride

    def generate(
        self,
        df,
        split_name="dataset"
    ):
        """
        Generate sliding windows independently for each engine.

        Parameters
        ----------
        df : pandas.DataFrame
            Dataset containing engine_id, cycle and RUL.

        split_name : str
            Name used when saving metadata.
            Examples: train, validation, test.

        Returns
        -------
        X : numpy.ndarray
            Shape:
            (samples, window_size, features)

        y : numpy.ndarray
            RUL target of the final timestep of each window.

        metadata : pandas.DataFrame
            Window-level metadata.
        """

        print("\n" + "=" * 70)
        print(f"SLIDING WINDOW GENERATION - {split_name.upper()}")
        print("=" * 70)

        # --------------------------------------------------
        # Features used inside each window
        # --------------------------------------------------

        excluded_columns = {
            "engine_id",
            "cycle",
            "RUL",
            "max_z_score",
            "zscore_flag"
        }

        feature_columns = [
            col
            for col in df.columns
            if col not in excluded_columns
        ]

        # --------------------------------------------------
        # Validation
        # --------------------------------------------------

        required_columns = {
            "engine_id",
            "cycle",
            "RUL"
        }

        missing_columns = (
            required_columns
            - set(df.columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if len(feature_columns) == 0:
            raise ValueError(
                "No feature columns available "
                "for window generation."
            )

        # --------------------------------------------------
        # Storage
        # --------------------------------------------------

        X = []
        y = []
        metadata_rows = []

        window_id = 0

        engines = sorted(
            df["engine_id"].unique()
        )

        print(
            f"Engines      : {len(engines)}"
        )

        print(
            f"Window Size  : {self.window_size}"
        )

        print(
            f"Stride       : {self.stride}"
        )

        print(
            f"Features     : {len(feature_columns)}"
        )

        # --------------------------------------------------
        # Generate windows engine by engine
        # --------------------------------------------------

        for engine_id in engines:

            engine_df = (
                df[
                    df["engine_id"] == engine_id
                ]
                .sort_values("cycle")
                .reset_index(drop=True)
            )

            total_cycles = len(engine_df)

            if total_cycles < self.window_size:
                continue

            for start in range(
                0,
                total_cycles - self.window_size + 1,
                self.stride
            ):

                end = (
                    start
                    + self.window_size
                )

                window = engine_df.iloc[
                    start:end
                ]

                # ------------------------------------------
                # Input sequence
                # ------------------------------------------

                X.append(
                    window[
                        feature_columns
                    ].to_numpy(
                        dtype=np.float32
                    )
                )

                # ------------------------------------------
                # Target = RUL at final timestep
                # ------------------------------------------

                target_rul = float(
                    window["RUL"].iloc[-1]
                )

                y.append(
                    target_rul
                )

                # ------------------------------------------
                # Metadata
                # ------------------------------------------

                metadata_rows.append(
                    {
                        "window_id": window_id,
                        "engine_id": int(engine_id),
                        "start_cycle": int(
                            window["cycle"].iloc[0]
                        ),
                        "end_cycle": int(
                            window["cycle"].iloc[-1]
                        ),
                        "target_rul": target_rul,
                    }
                )

                window_id += 1

        # --------------------------------------------------
        # Convert to NumPy
        # --------------------------------------------------

        if len(X) == 0:
            raise ValueError(
                "No windows were generated. "
                "Check window size and dataset length."
            )

        X = np.asarray(
            X,
            dtype=np.float32
        )

        y = np.asarray(
            y,
            dtype=np.float32
        )

        metadata = pd.DataFrame(
            metadata_rows
        )

        # --------------------------------------------------
        # Save metadata
        # --------------------------------------------------

        TABLES_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        metadata_file = (
            TABLES_DIR
            / f"window_information_{split_name}.csv"
        )

        metadata.to_csv(
            metadata_file,
            index=False
        )

        # --------------------------------------------------
        # Output information
        # --------------------------------------------------

        print(
            f"\n✓ Window metadata saved:"
        )

        print(
            f"  {metadata_file}"
        )

        print(
            f"\nTotal Windows : {len(X)}"
        )

        print(
            f"Feature Shape : {X.shape}"
        )

        print(
            f"Target Shape  : {y.shape}"
        )

        print(
            "\nFirst Window Shape:"
        )

        print(
            X[0].shape
        )

        print(
            "\nFirst Window Target:"
        )

        print(
            y[0]
        )

        print(
            "\nFirst Feature Vector:"
        )

        print(
            X[0][0]
        )

        return X, y, metadata