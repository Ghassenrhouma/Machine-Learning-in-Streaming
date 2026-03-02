"""
Train a fraud detection model on the Credit Card Fraud dataset.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Place creditcard.csv in the data/ directory before running.

Outputs:
  - ml/fraud_model.joblib      : trained RandomForest classifier
  - ml/baseline_stats.joblib   : per-feature mean/std for anomaly detection
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "fraud_model.joblib")
STATS_OUT = os.path.join(os.path.dirname(__file__), "baseline_stats.joblib")

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COL = "Class"

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[ERROR] Dataset not found at: {path}")
        print("Please download creditcard.csv from Kaggle and place it in data/")
        sys.exit(1)
    print(f"[INFO] Loading dataset from {path} ...")
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df):,} rows — fraud rate: {df[TARGET_COL].mean():.4%}")
    return df


def compute_baseline_stats(df: pd.DataFrame) -> dict:
    """Compute per-feature mean/std on the *normal* class for anomaly detection."""
    normal = df[df[TARGET_COL] == 0][FEATURE_COLS]
    stats = {
        "mean": normal.mean().to_dict(),
        "std": normal.std().to_dict(),
        "feature_cols": FEATURE_COLS,
        "baseline_fraud_rate": float(df[TARGET_COL].mean()),  # used by monitoring.py
    }
    return stats


def train(df: pd.DataFrame):
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Build pipeline: scale Amount (V-features are already PCA-scaled)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=100,
                    class_weight="balanced",     # handle class imbalance
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("[INFO] Training RandomForest ...")
    pipeline.fit(X_train, y_train)

    # ── Evaluation ──
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n[RESULTS] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    print(f"[RESULTS] ROC-AUC : {roc_auc_score(y_test, y_prob):.4f}")
    print(f"[RESULTS] Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}\n")

    return pipeline


def main():
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)

    df = load_data(DATA_PATH)

    # Baseline statistics (used by anomaly_detection.py)
    stats = compute_baseline_stats(df)
    joblib.dump(stats, STATS_OUT)
    print(f"[INFO] Baseline stats saved → {STATS_OUT}")

    # Train model
    model = train(df)
    joblib.dump(model, MODEL_OUT)
    print(f"[INFO] Model saved → {MODEL_OUT}")


if __name__ == "__main__":
    main()
