import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from xgboost import XGBClassifier

from ml_engine.generate_dataset import generate_ecommerce_dataset
from ml_engine.features import (
    compute_leakage_safe_features,
    temporal_train_val_test_split,
    build_preprocessor,
    ALL_FEATURE_COLS
)

def train_models(
    data_path: str = "ml_engine/data/ecommerce_orders.csv",
    artifact_dir: str = "ml_engine/artifacts"
):
    os.makedirs(artifact_dir, exist_ok=True)
    
    # 1. Load or Generate Dataset
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}, generating synthetic dataset...")
        df, _ = generate_ecommerce_dataset(output_dir="ml_engine/data", artifact_dir=artifact_dir)
    else:
        print(f"Loading dataset from {data_path}...")
        df = pd.read_csv(data_path)
        df['order_timestamp'] = pd.to_datetime(df['order_timestamp'])

    # 2. Compute Leakage-Safe Features
    print("Computing expanding window temporal features (leakage-safe)...")
    df_features = compute_leakage_safe_features(df)

    # 3. Temporal Train / Val / Test Split
    train_df, val_df, test_df = temporal_train_val_test_split(df_features, train_ratio=0.70, val_ratio=0.15)
    print(f"Split sizes - Train: {len(train_df)} | Val: {len(val_df)} | Held-Out Test: {len(test_df)}")

    X_train = train_df[ALL_FEATURE_COLS]
    y_train = train_df['is_returned'].values

    X_val = val_df[ALL_FEATURE_COLS]
    y_val = val_df['is_returned'].values

    X_test = test_df[ALL_FEATURE_COLS]
    y_test = test_df['is_returned'].values

    # 4. Fit Preprocessor on Train only
    print("Fitting ColumnTransformer on training split...")
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # 5. Train Baseline: Logistic Regression
    print("Training Baseline Logistic Regression...")
    baseline_lr = LogisticRegression(max_iter=1000, random_state=42)
    baseline_lr.fit(X_train_trans, y_train)

    # 6. Train Primary Model: XGBoost
    print("Training XGBoost Classifier...")
    xgb_base = XGBClassifier(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_base.fit(
        X_train_trans,
        y_train,
        eval_set=[(X_val_trans, y_val)],
        verbose=False
    )

    # 7. Probability Calibration on Validation Split (Isotonic with FrozenEstimator)
    print("Calibrating XGBoost probabilities on validation set (CalibratedClassifierCV)...")
    calibrated_xgb = CalibratedClassifierCV(
        estimator=FrozenEstimator(xgb_base),
        method='isotonic'
    )
    calibrated_xgb.fit(X_val_trans, y_val)

    # Save Preprocessor and Models
    joblib.dump(preprocessor, os.path.join(artifact_dir, "preprocessor.joblib"))
    joblib.dump(baseline_lr, os.path.join(artifact_dir, "baseline_lr.joblib"))
    joblib.dump(xgb_base, os.path.join(artifact_dir, "xgb_uncalibrated.joblib"))
    joblib.dump(calibrated_xgb, os.path.join(artifact_dir, "model.joblib"))

    # Save processed split DataFrames using joblib for fast reliable serialization
    joblib.dump(train_df, os.path.join(artifact_dir, "train_split.joblib"))
    joblib.dump(val_df, os.path.join(artifact_dir, "val_split.joblib"))
    joblib.dump(test_df, os.path.join(artifact_dir, "test_split.joblib"))

    print(f"Model artifacts successfully written to {artifact_dir}")
    return calibrated_xgb, preprocessor

if __name__ == "__main__":
    train_models()
