import pytest
import numpy as np
import pandas as pd
from ml_engine.generate_dataset import generate_ecommerce_dataset
from ml_engine.features import compute_leakage_safe_features, temporal_train_val_test_split, build_preprocessor, ALL_FEATURE_COLS

def test_dataset_generation_and_features():
    df, datacard = generate_ecommerce_dataset(n_orders=500, n_customers=100, noise_rate=0.10, seed=123)
    assert len(df) == 500
    assert 'is_returned' in df.columns
    assert datacard['injected_label_noise_rate'] == 0.10
    
    df_feat = compute_leakage_safe_features(df)
    for col in ALL_FEATURE_COLS:
        assert col in df_feat.columns, f"Missing feature {col}"
    
    train_df, val_df, test_df = temporal_train_val_test_split(df_feat)
    assert len(train_df) == 350
    assert len(val_df) == 75
    assert len(test_df) == 75
    
    # Assert strict temporal ordering
    assert train_df['order_timestamp'].max() <= val_df['order_timestamp'].min()
    assert val_df['order_timestamp'].max() <= test_df['order_timestamp'].min()

def test_preprocessor_transformation():
    df, _ = generate_ecommerce_dataset(n_orders=200, n_customers=50, noise_rate=0.10, seed=42)
    df_feat = compute_leakage_safe_features(df)
    preprocessor = build_preprocessor()
    X = preprocessor.fit_transform(df_feat[ALL_FEATURE_COLS])
    assert X.shape[0] == 200
    assert X.shape[1] > len(ALL_FEATURE_COLS) # Due to one-hot encoding
    assert not np.isnan(X).any(), "NaN values found in preprocessed matrix"
