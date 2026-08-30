from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ActionEnum(str, Enum):
    ALLOW_COD = "ALLOW_COD"
    NUDGE_UPI_CASHBACK = "NUDGE_UPI_CASHBACK"
    BLOCK_COD_REQUIRE_PREPAID = "BLOCK_COD_REQUIRE_PREPAID"

class RiskTierEnum(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

class DriverItem(BaseModel):
    feature_name: str
    display_name: str
    direction: str = Field(description="'INCREASES_RISK' or 'REDUCES_RISK'")
    impact_score: float
    explanation: str

class OrderScoreRequest(BaseModel):
    order_id: Optional[str] = Field(default="ORD_DEMO_001", description="Unique transaction ID")
    customer_id: str = Field(default="CUST_004821", description="Unique customer ID")
    order_amount: float = Field(default=1899.0, ge=1.0, description="Order total in INR")
    discount_percentage: float = Field(default=0.25, ge=0.0, le=1.0, description="Discount fraction e.g. 0.25 for 25%")
    product_category: str = Field(default="Apparel", description="Apparel, Footwear, Electronics, Beauty_PersonalCare, Home_Kitchen")
    item_quantity: int = Field(default=1, ge=1, description="Number of items in the cart")
    payment_method: str = Field(default="COD", description="COD, UPI, CreditCard, DebitCard, NetBanking")
    pincode_tier: str = Field(default="Tier2_Urban", description="Tier1_Metro, Tier2_Urban, Tier3_SemiUrban, Remote")
    shipping_speed: str = Field(default="Standard", description="Standard, Express")
    device_type: str = Field(default="Mobile_App", description="Mobile_App, Mobile_Web, Desktop")
    user_past_orders_count: Optional[int] = Field(default=0, ge=0, description="Historical completed orders by customer")
    user_past_return_rate: Optional[float] = Field(default=0.20, ge=0.0, le=1.0, description="Customer past return rate")

class OrderScoreResponse(BaseModel):
    order_id: str
    risk_score: float = Field(description="Calibrated probability of return/RTO [0.0 - 1.0]")
    risk_tier: RiskTierEnum
    action: ActionEnum
    action_reason: str
    potential_savings_inr: float = Field(description="Deterministic savings: C_fn * p for BLOCK_COD; C_fn * p * 0.35 for NUDGE_UPI; 0 for ALLOW_COD")
    top_drivers: List[DriverItem]
    thresholds_applied: Dict[str, float]
    evaluated_at: str

class MetricsResponse(BaseModel):
    model_name: str
    baseline_name: str
    optimal_threshold: float
    low_threshold: float
    cost_assumptions: Dict[str, Any]
    held_out_test_metrics: Dict[str, Any]
    baseline_comparison: Dict[str, Any]
    validation_threshold_sweep: List[Dict[str, Any]]
    pr_curve: List[Dict[str, float]]
    roc_curve: List[Dict[str, float]]
    top_global_features: List[Dict[str, Any]]
    datacard: Dict[str, Any]
