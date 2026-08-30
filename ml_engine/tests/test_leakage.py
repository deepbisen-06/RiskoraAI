import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from ml_engine.features import compute_leakage_safe_features

def test_no_future_leakage_in_aggregates():
    """
    Test that altering future orders has ZERO impact on feature values of earlier orders.
    """
    start_time = datetime(2025, 1, 1, 10, 0, 0)
    
    # Create 4 orders for the same customer over time
    base_data = [
        {"order_id": "ORD_1", "customer_id": "CUST_A", "order_timestamp": start_time, "order_amount": 1000, "discount_percentage": 0.1, "product_category": "Apparel", "item_quantity": 1, "payment_method": "COD", "pincode_tier": "Tier1_Metro", "shipping_speed": "Standard", "device_type": "Mobile_App", "is_returned": 0},
        {"order_id": "ORD_2", "customer_id": "CUST_A", "order_timestamp": start_time + timedelta(days=1), "order_amount": 1200, "discount_percentage": 0.2, "product_category": "Apparel", "item_quantity": 1, "payment_method": "UPI", "pincode_tier": "Tier1_Metro", "shipping_speed": "Standard", "device_type": "Mobile_App", "is_returned": 1},
        {"order_id": "ORD_3", "customer_id": "CUST_A", "order_timestamp": start_time + timedelta(days=2), "order_amount": 1500, "discount_percentage": 0.05, "product_category": "Electronics", "item_quantity": 1, "payment_method": "COD", "pincode_tier": "Tier1_Metro", "shipping_speed": "Standard", "device_type": "Mobile_App", "is_returned": 0},
        {"order_id": "ORD_4", "customer_id": "CUST_A", "order_timestamp": start_time + timedelta(days=3), "order_amount": 2000, "discount_percentage": 0.3, "product_category": "Footwear", "item_quantity": 2, "payment_method": "COD", "pincode_tier": "Tier1_Metro", "shipping_speed": "Standard", "device_type": "Mobile_App", "is_returned": 1},
    ]
    
    df1 = pd.DataFrame(base_data)
    res1 = compute_leakage_safe_features(df1)
    
    # Modify future row ORD_4's return status from 1 to 0 and change amount
    modified_data = [d.copy() for d in base_data]
    modified_data[3]["is_returned"] = 0
    modified_data[3]["order_amount"] = 99999
    
    df2 = pd.DataFrame(modified_data)
    res2 = compute_leakage_safe_features(df2)
    
    # Assert that ORD_1, ORD_2, ORD_3 features in res1 are EXACTLY identical to res2
    for col in ['user_past_orders_count', 'user_past_return_rate', 'pincode_historical_rto_rate']:
        np.testing.assert_array_almost_equal(
            res1.iloc[:3][col].values,
            res2.iloc[:3][col].values,
            err_msg=f"Feature {col} leaked future data from modified row ORD_4!"
        )

def test_first_order_cold_start():
    """
    Test that a customer's first order has past_orders_count == 0.
    """
    start_time = datetime(2025, 1, 1, 10, 0, 0)
    data = [
        {"order_id": "ORD_1", "customer_id": "CUST_NEW", "order_timestamp": start_time, "order_amount": 1000, "discount_percentage": 0.1, "product_category": "Apparel", "item_quantity": 1, "payment_method": "COD", "pincode_tier": "Tier1_Metro", "shipping_speed": "Standard", "device_type": "Mobile_App", "is_returned": 1}
    ]
    res = compute_leakage_safe_features(pd.DataFrame(data))
    assert res.iloc[0]['user_past_orders_count'] == 0
    assert res.iloc[0]['is_first_time_buyer_cod'] == 1.0
