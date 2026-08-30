import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_score_order_endpoint_high_risk():
    payload = {
        "order_id": "TEST_ORD_001",
        "customer_id": "CUST_TEST_HIGH",
        "order_amount": 3499.0,
        "discount_percentage": 0.45,
        "product_category": "Apparel",
        "item_quantity": 2,
        "payment_method": "COD",
        "pincode_tier": "Remote",
        "shipping_speed": "Standard",
        "device_type": "Mobile_Web",
        "user_past_orders_count": 4,
        "user_past_return_rate": 0.80
    }
    response = client.post("/api/v1/score-order", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "risk_tier" in data
    assert "action" in data
    assert "potential_savings_inr" in data
    assert len(data["top_drivers"]) >= 1
    assert data["risk_tier"] in ["LOW", "MEDIUM", "HIGH"]
    assert data["action"] in ["ALLOW_COD", "NUDGE_UPI_CASHBACK", "BLOCK_COD_REQUIRE_PREPAID"]

def test_score_order_endpoint_low_risk():
    payload = {
        "order_id": "TEST_ORD_002",
        "customer_id": "CUST_TEST_VIP",
        "order_amount": 4500.0,
        "discount_percentage": 0.05,
        "product_category": "Electronics",
        "item_quantity": 1,
        "payment_method": "UPI",
        "pincode_tier": "Tier1_Metro",
        "shipping_speed": "Express",
        "device_type": "Mobile_App",
        "user_past_orders_count": 15,
        "user_past_return_rate": 0.00
    }
    response = client.post("/api/v1/score-order", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_tier"] == "LOW"
    assert data["action"] == "ALLOW_COD"
    assert data["potential_savings_inr"] == 0.0
