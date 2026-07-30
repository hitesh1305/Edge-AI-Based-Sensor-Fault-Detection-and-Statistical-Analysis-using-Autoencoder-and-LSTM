import os
import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def train_isolation_forest(feature_df, labels):

    print("\nTraining Isolation Forest...")

    # -----------------------------------------
    # Standardize Features
    # -----------------------------------------

    scaler = StandardScaler()

    X = scaler.fit_transform(feature_df)

    contamination_values = [
        0.05,
        0.08,
        0.09,
        0.10,
        0.12,
        0.15
    ]

    results = []

    best_model = None
    best_predictions = None
    best_f1 = -1
    best_contamination = None

    # -----------------------------------------
    # Grid Search
    # -----------------------------------------

    for contamination in contamination_values:

        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            random_state=42
        )

        model.fit(X)

        predictions = model.predict(X)

        predictions = (predictions == -1).astype(int)

        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)

        results.append([
            contamination,
            accuracy,
            precision,
            recall,
            f1
        ])

        if f1 > best_f1:

            best_f1 = f1
            best_model = model
            best_predictions = predictions
            best_contamination = contamination

    # -----------------------------------------
    # Print Comparison
    # -----------------------------------------

    results_df = pd.DataFrame(
        results,
        columns=[
            "Contamination",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    )

    print("\nGrid Search Results\n")
    print(results_df)

    # -----------------------------------------
    # Best Model
    # -----------------------------------------

    print("\nBest Contamination :", best_contamination)

    print("\nFinal Metrics")

    print(f"Accuracy  : {accuracy_score(labels,best_predictions):.4f}")
    print(f"Precision : {precision_score(labels,best_predictions):.4f}")
    print(f"Recall    : {recall_score(labels,best_predictions):.4f}")
    print(f"F1 Score  : {f1_score(labels,best_predictions):.4f}")

    print("\nConfusion Matrix")

    print(confusion_matrix(labels,best_predictions))

    print("\nClassification Report")

    print(classification_report(labels,best_predictions))

    # -----------------------------------------
    # Save Everything
    # -----------------------------------------

    os.makedirs("output", exist_ok=True)
    os.makedirs("models/saved_models", exist_ok=True)

    results_df.to_csv(
        "output/isolation_forest_grid_search.csv",
        index=False
    )

    prediction_df = feature_df.copy()

    prediction_df["Actual"] = labels
    prediction_df["Predicted"] = best_predictions

    prediction_df.to_csv(
        "output/isolation_forest_predictions.csv",
        index=False
    )

    joblib.dump(
        best_model,
        "models/saved_models/isolation_forest.pkl"
    )

    joblib.dump(
        scaler,
        "models/saved_models/isolation_scaler.pkl"
    )

    print("\n✓ Isolation Forest Completed")

    return best_model