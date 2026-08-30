import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_ecommerce_dataset(
    n_orders: int = 60000,
    n_customers: int = 12000,
    noise_rate: float = 0.10,
    seed: int = 42,
    output_dir: str = "ml_engine/data",
    artifact_dir: str = "ml_engine/artifacts"
):
    """
    Generates realistic chronological e-commerce order logs with intrinsic customer behaviors,
    category propensities, and explicit label noise injection.
    """
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(artifact_dir, exist_ok=True)

    print(f"Generating {n_orders} sequential e-commerce orders across {n_customers} customers...")

    # 1. Customer base with intrinsic latent return propensities
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]
    # Latent return propensity: Beta distribution
    customer_latent_propensity = dict(zip(customer_ids, np.random.beta(a=1.2, b=5.0, size=n_customers)))

    # 2. Timeline generation over 365 days
    start_date = datetime(2025, 1, 1, 0, 0, 0)
    seconds_spread = np.sort(np.random.uniform(0, 365 * 24 * 3600, size=n_orders))
    order_timestamps = [start_date + timedelta(seconds=int(s)) for s in seconds_spread]

    categories = ['Apparel', 'Footwear', 'Electronics', 'Beauty_PersonalCare', 'Home_Kitchen']
    cat_weights = [0.35, 0.20, 0.18, 0.15, 0.12]
    cat_base_risk = {
        'Apparel': 0.38,
        'Footwear': 0.30,
        'Home_Kitchen': 0.14,
        'Beauty_PersonalCare': 0.08,
        'Electronics': 0.05
    }

    payment_methods = ['COD', 'UPI', 'CreditCard', 'DebitCard', 'NetBanking']
    pay_weights = [0.42, 0.32, 0.14, 0.08, 0.04]

    pincode_tiers = ['Tier1_Metro', 'Tier2_Urban', 'Tier3_SemiUrban', 'Remote']
    pincode_weights = [0.45, 0.30, 0.18, 0.07]
    pincode_risk_mod = {
        'Tier1_Metro': -0.10,
        'Tier2_Urban': 0.00,
        'Tier3_SemiUrban': 0.12,
        'Remote': 0.25
    }

    devices = ['Mobile_App', 'Mobile_Web', 'Desktop']
    device_weights = [0.65, 0.25, 0.10]

    # Sample attributes
    assigned_customers = np.random.choice(customer_ids, size=n_orders)
    assigned_categories = np.random.choice(categories, size=n_orders, p=cat_weights)
    assigned_payments = np.random.choice(payment_methods, size=n_orders, p=pay_weights)
    assigned_pincodes = np.random.choice(pincode_tiers, size=n_orders, p=pincode_weights)
    assigned_devices = np.random.choice(devices, size=n_orders, p=device_weights)
    assigned_shipping = np.random.choice(['Standard', 'Express'], size=n_orders, p=[0.75, 0.25])
    assigned_quantities = np.random.choice([1, 2, 3, 4, 5], size=n_orders, p=[0.68, 0.20, 0.07, 0.03, 0.02])

    category_mean_price = {
        'Apparel': 1499,
        'Footwear': 2499,
        'Electronics': 4999,
        'Beauty_PersonalCare': 799,
        'Home_Kitchen': 1899
    }
    
    order_amounts = []
    discounts = []
    
    for cat, qty in zip(assigned_categories, assigned_quantities):
        mean_p = category_mean_price[cat]
        unit_price = max(199, np.random.lognormal(mean=np.log(mean_p), sigma=0.45))
        disc = np.clip(np.random.beta(a=2.0, b=5.0), 0.0, 0.70)
        final_amount = round(unit_price * qty * (1.0 - disc), 2)
        order_amounts.append(final_amount)
        discounts.append(round(disc, 4))

    # Calculate ground truth return probabilities
    true_probabilities = []
    clean_labels = []

    for i in range(n_orders):
        cust = assigned_customers[i]
        cat = assigned_categories[i]
        pay = assigned_payments[i]
        pin = assigned_pincodes[i]
        disc = discounts[i]
        qty = assigned_quantities[i]
        ship = assigned_shipping[i]

        cust_risk = customer_latent_propensity[cust]
        cat_risk = cat_base_risk[cat]
        pin_risk = pincode_risk_mod[pin]
        pay_risk = 0.35 if pay == 'COD' else -0.25
        disc_risk = 0.25 * (disc / 0.70)
        size_risk = 0.12 if (cat in ['Apparel', 'Footwear'] and qty > 1) else 0.0
        ship_risk = -0.06 if ship == 'Express' else 0.03

        # Combine into log-odds centered so average return rate is ~22%
        logit = (
            -1.6 +
            2.5 * cust_risk +
            2.0 * (cat_risk - 0.20) +
            1.8 * pay_risk +
            1.4 * pin_risk +
            1.5 * disc_risk +
            1.2 * size_risk +
            ship_risk
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        true_probabilities.append(prob)
        clean_labels.append(1 if np.random.uniform(0, 1) < prob else 0)

    # 3. Explicit label noise injection (flips labels at configured 10% rate)
    noise_mask = np.random.binomial(1, noise_rate, size=n_orders).astype(bool)
    noisy_labels = np.array(clean_labels)
    noisy_labels[noise_mask] = 1 - noisy_labels[noise_mask]

    # 4. Construct DataFrame
    df = pd.DataFrame({
        'order_id': [f"ORD_{i:07d}" for i in range(1, n_orders + 1)],
        'customer_id': assigned_customers,
        'order_timestamp': order_timestamps,
        'order_amount': order_amounts,
        'discount_percentage': discounts,
        'product_category': assigned_categories,
        'item_quantity': assigned_quantities,
        'payment_method': assigned_payments,
        'pincode_tier': assigned_pincodes,
        'shipping_speed': assigned_shipping,
        'device_type': assigned_devices,
        'true_return_prob': np.round(true_probabilities, 4),
        'is_returned': noisy_labels.tolist()
    })

    # Sort strictly by timestamp to maintain temporal integrity
    df = df.sort_values('order_timestamp').reset_index(drop=True)

    csv_path = os.path.join(output_dir, "ecommerce_orders.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved dataset to {csv_path} ({len(df)} rows)")

    # 5. Export Datacard
    return_rate = float(df['is_returned'].mean())
    datacard = {
        "dataset_name": "E-Commerce Returns & RTO Benchmark (Temporal Log)",
        "total_orders": n_orders,
        "unique_customers": n_customers,
        "date_range": {
            "start": str(df['order_timestamp'].min()),
            "end": str(df['order_timestamp'].max())
        },
        "return_rate": round(return_rate, 4),
        "injected_label_noise_rate": noise_rate,
        "temporal_split_ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
        "features": {
            "categorical": ["product_category", "payment_method", "pincode_tier", "shipping_speed", "device_type"],
            "numerical": ["order_amount", "discount_percentage", "item_quantity"],
            "temporal_aggregates": ["user_past_orders_count", "user_past_return_rate", "pincode_historical_rto_rate"],
            "derived": ["discount_depth_amount", "price_per_item", "is_first_time_buyer_cod"]
        },
        "category_distribution": df['product_category'].value_counts(normalize=True).to_dict(),
        "payment_distribution": df['payment_method'].value_counts(normalize=True).to_dict()
    }

    datacard_path = os.path.join(artifact_dir, "datacard.json")
    with open(datacard_path, "w", encoding="utf-8") as f:
        json.dump(datacard, f, indent=2)
    print(f"Saved datacard to {datacard_path}")

    return df, datacard

if __name__ == "__main__":
    generate_ecommerce_dataset()
