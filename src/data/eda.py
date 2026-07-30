"""
eda.py
-------

Exploratory Data Analysis for NASA C-MAPSS
"""

import os
import pandas as pd


class EDA:

    def __init__(self, df):
        self.df = df.copy()

    def dataset_info(self):

        print("\n" + "=" * 70)
        print("DATASET INFORMATION")
        print("=" * 70)

        print(f"Rows      : {self.df.shape[0]}")
        print(f"Columns   : {self.df.shape[1]}")

        print(f"Engines   : {self.df['engine_id'].nunique()}")

        print(f"Max Cycle : {self.df['cycle'].max()}")

    def missing_values(self):

        print("\n" + "=" * 70)
        print("MISSING VALUES")
        print("=" * 70)

        missing = self.df.isnull().sum()

        print(missing[missing > 0])

    def duplicate_rows(self):

        print("\n" + "=" * 70)
        print("DUPLICATE ROWS")
        print("=" * 70)

        print(self.df.duplicated().sum())

    def constant_columns(self):

        print("\n" + "=" * 70)
        print("CONSTANT / LOW VARIANCE COLUMNS")
        print("=" * 70)

        nunique = self.df.nunique()

        constant = nunique[nunique == 1]

        if len(constant) == 0:
            print("No completely constant columns.")
        else:
            print(constant)

    def sensor_statistics(self):

        print("\n" + "=" * 70)
        print("SENSOR SUMMARY")
        print("=" * 70)

        sensors = [c for c in self.df.columns if "sensor_" in c]

        print(
            self.df[sensors].describe().T[
                ["mean", "std", "min", "max"]
            ]
        )

    def save_summary(self, output_path="output"):

        os.makedirs(output_path, exist_ok=True)

        summary = self.df.describe(include="all")

        summary.to_csv(f"{output_path}/eda_summary.csv")

        print("\nEDA summary saved.")