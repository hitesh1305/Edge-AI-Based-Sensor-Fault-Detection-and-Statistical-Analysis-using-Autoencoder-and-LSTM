import os
import pandas as pd

from preprocessing.cleaning import load_dataset, clean_dataset
from preprocessing.normalize import normalize_features
from preprocessing.sliding_window import create_sequences

from analysis.statistical_analysis import (
    statistical_analysis,
    z_score_analysis,
    generate_plots
)

from filters.ekf_filter import apply_ekf


from feature_extraction.wavelet_transform import apply_wavelet_transform

from feature_extraction.window_features import extract_window_features

from models.isolation_forest import train_isolation_forest

# ==========================================================
# Dataset Path
# ==========================================================

DATASET = "datasets/smart_manufacturing_data.csv"

# Create output folder
os.makedirs("output", exist_ok=True)

# ==========================================================
# Step 1 : Load Dataset
# ==========================================================

df = load_dataset(DATASET)

# ==========================================================
# Step 2 : Clean Dataset
# ==========================================================

df = clean_dataset(df)

print("\n✓ Cleaning Completed")

# ==========================================================
# Step 3 : Apply EKF
# ==========================================================

filtered_df = apply_ekf(df)

filtered_df.to_csv(
    "output/filtered_data.csv",
    index=False
)

print("✓ EKF Filtering Completed")

# ==========================================================
# Step 4 : Statistical Analysis
# ==========================================================

statistical_analysis(filtered_df)

filtered_df = z_score_analysis(filtered_df)

print("\nAnomaly Distribution")
print(filtered_df["anomaly_flag"].value_counts())

print("\nFailure Type Distribution")
print(filtered_df["failure_type"].value_counts())

print("\nAnomaly Flag vs Failure Type")
print(pd.crosstab(
    filtered_df["anomaly_flag"],
    filtered_df["failure_type"]
))

generate_plots(filtered_df)

print("✓ Statistical Analysis Completed")

# Save dataset with Z-score labels
filtered_df.to_csv(
    "output/statistical_dataset.csv",
    index=False
)

# ==========================================================
# Step 5 : Normalize Features
# ==========================================================

normalized_df, scaler = normalize_features(filtered_df)

normalized_df.to_csv(
    "output/normalized_data.csv",
    index=False
)

print("✓ Normalization Completed")

# ==========================================================
# Step 6 : Sliding Window
# ==========================================================

X = create_sequences(normalized_df)

print("\n✓ Sliding Window Completed")
print("Sliding Window Shape:", X.shape)

print("\nFirst Window:")
print(X[0])


print("\n========== PREPROCESSING PIPELINE COMPLETED ==========")

# ==========================================================
# Step 7 : Wavelet Transform
# ==========================================================

wavelet_result = apply_wavelet_transform(filtered_df)

wavelet_features = wavelet_result["features"]

print("\n✓ Wavelet Transform Completed")

# ==========================================================
# Step 8 : Window Feature Extraction
# ==========================================================

window_feature_df = extract_window_features(X)

print(window_feature_df.head())
print(window_feature_df.describe())

# ==========================================================
# Step 9 : Isolation Forest
# ==========================================================

WINDOW_SIZE = 20

X = create_sequences(normalized_df, window_size=WINDOW_SIZE)

labels = (
    filtered_df["anomaly_flag"]
    .iloc[WINDOW_SIZE:]
    .reset_index(drop=True)
)

isolation_model = train_isolation_forest(
    window_feature_df,
    labels
)

print("\n========== ISOLATION FOREST COMPLETED ==========")