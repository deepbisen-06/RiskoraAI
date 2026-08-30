import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = [
    'product_category',
    'payment_method',
    'pincode_tier',
    'shipping_speed',
    'device_type'
]

NUMERICAL_FEATURES = [
    'order_amount',
    'discount_percentage',
    'item_quantity',
    'user_past_orders_count',
    'user_past_return_rate',
    'pincode_historical_rto_rate',
    'discount_depth_amount',
    'price_per_item',
    'is_first_time_buyer_cod'
]

ALL_FEATURE_COLS = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

def compute_leakage_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes expanding-window customer and pincode return statistics strictly using
    prior rows (order_timestamp < current order_timestamp).
    No information from the current order or future orders is used.
    """
    # Ensure sorted by timestamp
    df = df.sort_values('order_timestamp').copy()
    
    # 1. Customer Expanding Statistics
    # We compute cumulative count and cumulative sum of returns per customer strictly up to previous row (shift(1))
    grouped_cust = df.groupby('customer_id')
    
    # Shift(1) ensures that the current row's is_returned outcome is NOT included
    past_orders = grouped_cust.cumcount() # 0 for first order, 1 for second, etc.
    past_returns = grouped_cust['is_returned'].transform(lambda s: s.shift(1).fillna(0).cumsum())
    
    df['user_past_orders_count'] = past_orders.values
    # Laplace smoothed return rate: (returns + 1) / (orders + 5) for smooth Bayesian cold-start prior
    df['user_past_return_rate'] = (
        (past_returns + 1.0) / (past_orders + 5.0)
    ).values

    # 2. Pincode Tier Historical RTO Rate (expanding window with shift)
    grouped_pin = df.groupby('pincode_tier')
    pin_past_orders = grouped_pin.cumcount()
    pin_past_returns = grouped_pin['is_returned'].transform(lambda s: s.shift(1).fillna(0).cumsum())
    df['pincode_historical_rto_rate'] = (
        (pin_past_returns + 1.0) / (pin_past_orders + 10.0)
    ).values

    # 3. Derived Interaction Features
    df['discount_depth_amount'] = df['order_amount'] * df['discount_percentage']
    df['price_per_item'] = df['order_amount'] / np.maximum(df['item_quantity'], 1)
    df['is_first_time_buyer_cod'] = ((df['user_past_orders_count'] == 0) & (df['payment_method'] == 'COD')).astype(float)

    return df

def temporal_train_val_test_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """
    Splits chronologically sorted dataset into Train (70%), Validation (15%), and Held-out Test (15%).
    """
    df_sorted = df.sort_values('order_timestamp').reset_index(drop=True)
    n = len(df_sorted)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_df = df_sorted.iloc[:train_end].copy()
    val_df = df_sorted.iloc[train_end:val_end].copy()
    test_df = df_sorted.iloc[val_end:].copy()
    
    return train_df, val_df, test_df

def build_preprocessor() -> ColumnTransformer:
    """
    Builds standard Scikit-Learn ColumnTransformer for categorical and numerical features.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor
