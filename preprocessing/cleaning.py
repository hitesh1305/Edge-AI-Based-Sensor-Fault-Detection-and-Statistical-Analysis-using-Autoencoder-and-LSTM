import os

import pandas as pd


def load_dataset(file_path):
    """
    Load CSV dataset
    """
    df = pd.read_csv(file_path)

    print("\nDataset Loaded Successfully")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    return df


def clean_dataset(df):
    """
    Perform basic cleaning
    """

    print("\nChecking Missing Values...")
    print(df.isnull().sum())

    print("\nRemoving Duplicate Rows...")
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    print(f"Duplicates Removed : {before-after}")


    os.makedirs("output", exist_ok=True)

    df.to_csv("output/cleaned_data.csv", index=False)

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df

