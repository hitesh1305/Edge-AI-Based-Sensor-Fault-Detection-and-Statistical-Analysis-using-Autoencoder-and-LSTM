import os
from statistics import correlation
from numpy.char import count
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import zscore
from scipy.stats import skew, kurtosis

FEATURE_COLUMNS = [
    "temperature",
    "vibration",
    "humidity",
    "pressure",
    "energy_consumption"
]


def statistical_analysis(df):

    report = pd.DataFrame(index=FEATURE_COLUMNS)

    report["Mean"] = df[FEATURE_COLUMNS].mean()
    report["Median"] = df[FEATURE_COLUMNS].median()
    report["Std Dev"] = df[FEATURE_COLUMNS].std()
    report["Variance"] = df[FEATURE_COLUMNS].var()
    report["Minimum"] = df[FEATURE_COLUMNS].min()
    report["Maximum"] = df[FEATURE_COLUMNS].max()

    report["Skewness"] = [
        skew(df[col]) for col in FEATURE_COLUMNS
    ]

    report["Kurtosis"] = [
        kurtosis(df[col]) for col in FEATURE_COLUMNS
    ]

    os.makedirs("output", exist_ok=True)

    report.to_csv("output/statistical_report.csv")

    print("\nStatistical Report")
    print(report)

    return report



def z_score_analysis(df):

    z_scores = df[FEATURE_COLUMNS].apply(zscore)

    threshold = 3

    anomalies = (z_scores.abs() > threshold).any(axis=1)

    count = anomalies.sum()
    percentage = (count / len(df)) * 100

    print(f"\nZ-Score Anomalies : {count}")
    print(f"Percentage         : {percentage:.2f}%")

    df["zscore_anomaly"] = anomalies.astype(int)

    df.to_csv("output/statistical_dataset.csv", index=False)

    return df

def generate_plots(df):

    os.makedirs("output/plots", exist_ok=True)

    for column in FEATURE_COLUMNS:

        plt.figure(figsize=(8,4))

        plt.hist(df[column], bins=30)

        plt.title(column)

        plt.tight_layout()

        plt.savefig(f"output/plots/{column}_histogram.png")

        plt.close()

    correlation = df[FEATURE_COLUMNS].corr()

    plt.figure(figsize=(6,5))

    plt.figure(figsize=(8,6))

    plt.imshow(correlation,
           cmap="coolwarm",
           vmin=-1,
           vmax=1)

    plt.colorbar()

    plt.xticks(range(len(FEATURE_COLUMNS)),
           FEATURE_COLUMNS,
           rotation=45)

    plt.yticks(range(len(FEATURE_COLUMNS)),
           FEATURE_COLUMNS)

    for i in range(len(FEATURE_COLUMNS)):
        for j in range(len(FEATURE_COLUMNS)):
            plt.text(j, i,
                    f"{correlation.iloc[i,j]:.2f}",
                    ha="center",
                    color="black")

    plt.tight_layout()
    
    plt.colorbar()

    plt.xticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS, rotation=45)
    plt.yticks(range(len(FEATURE_COLUMNS)), FEATURE_COLUMNS)

    plt.tight_layout()

    plt.savefig("output/plots/correlation_matrix.png")

    plt.close()