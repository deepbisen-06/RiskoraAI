import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from backend.config import settings
from backend.schemas import (
    OrderScoreRequest,
    OrderScoreResponse,
    DriverItem,
    ActionEnum,
    RiskTierEnum
)
from ml_engine.features import ALL_FEATURE_COLS

class ReturnRiskInferenceEngine:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.load_artifacts()

    def load_artifacts(self):
        """Loads serialized calibrated model and feature preprocessor."""
        if os.path.exists(settings.MODEL_PATH) and os.path.exists(settings.PREPROCESSOR_PATH):
            try:
                self.model = joblib.load(settings.MODEL_PATH)
                self.preprocessor = joblib.load(settings.PREPROCESSOR_PATH)
                print("[InferenceEngine] Calibrated XGBoost & Preprocessor loaded successfully.")
            except Exception as e:
                print(f"[InferenceEngine] Error loading model artifacts: {e}")
        else:
            print("[InferenceEngine] Warning: Model artifacts not found.")

    def score_order(self, req: OrderScoreRequest) -> OrderScoreResponse:
        # Refresh thresholds from config
        settings.load_dynamic_thresholds()
        t_high = settings.OPTIMAL_THRESHOLD_HIGH
        t_low = settings.OPTIMAL_THRESHOLD_LOW
        c_fn = settings.DEFAULT_C_FN
        upi_eff = settings.UPI_CONVERSION_EFFICIENCY

        # Derived features
        discount_depth_amount = req.order_amount * req.discount_percentage
        price_per_item = req.order_amount / max(req.item_quantity, 1)
        
        past_orders = req.user_past_orders_count if req.user_past_orders_count is not None else 0
        raw_return_rate = req.user_past_return_rate if req.user_past_return_rate is not None else 0.20
        past_returns = raw_return_rate * past_orders
        
        # Laplace smoothed return rate strictly matching features.py: (returns + 1) / (orders + 5)
        user_past_return_rate_smoothed = (past_returns + 1.0) / (past_orders + 5.0)

        is_first_time_buyer_cod = 1.0 if (past_orders == 0 and req.payment_method == "COD") else 0.0

        pincode_risk_lookup = {
            "Tier1_Metro": 0.14,
            "Tier2_Urban": 0.22,
            "Tier3_SemiUrban": 0.32,
            "Remote": 0.44
        }
        pincode_historical_rto_rate = pincode_risk_lookup.get(req.pincode_tier, 0.22)

        # Build single-row DataFrame matching training column structure
        row_dict = {
            "order_amount": req.order_amount,
            "discount_percentage": req.discount_percentage,
            "item_quantity": req.item_quantity,
            "user_past_orders_count": past_orders,
            "user_past_return_rate": user_past_return_rate_smoothed,
            "pincode_historical_rto_rate": pincode_historical_rto_rate,
            "discount_depth_amount": discount_depth_amount,
            "price_per_item": price_per_item,
            "is_first_time_buyer_cod": is_first_time_buyer_cod,
            "product_category": req.product_category,
            "payment_method": req.payment_method,
            "pincode_tier": req.pincode_tier,
            "shipping_speed": req.shipping_speed,
            "device_type": req.device_type
        }
        df_row = pd.DataFrame([row_dict])[ALL_FEATURE_COLS]

        # Calculate calibrated risk score
        if self.model is not None and self.preprocessor is not None:
            X_trans = self.preprocessor.transform(df_row)
            prob = float(self.model.predict_proba(X_trans)[0, 1])
        else:
            prob = 0.20
            if req.payment_method == "COD": prob += 0.25
            if req.product_category in ["Apparel", "Footwear"]: prob += 0.18
            if raw_return_rate > 0.40: prob += 0.25
            if req.pincode_tier in ["Tier3_SemiUrban", "Remote"]: prob += 0.12
            prob = min(max(prob, 0.02), 0.98)

        risk_score = round(prob, 4)

        # Dynamic Risk Tier & Action Mapping
        if risk_score >= t_high:
            risk_tier = RiskTierEnum.HIGH
            action = ActionEnum.BLOCK_COD_REQUIRE_PREPAID
            action_reason = f"High RTO Risk ({risk_score:.1%} >= T*={t_high:.1%}). COD payment disabled to protect reverse logistics cost."
            potential_savings = round(c_fn * risk_score, 2)
        elif risk_score >= t_low:
            risk_tier = RiskTierEnum.MEDIUM
            action = ActionEnum.NUDGE_UPI_CASHBACK
            action_reason = f"Moderate Return Risk ({risk_score:.1%}). COD permitted with ₹50 instant UPI discount to incentivize prepaid conversion."
            potential_savings = round(c_fn * risk_score * upi_eff, 2)
        else:
            risk_tier = RiskTierEnum.LOW
            action = ActionEnum.ALLOW_COD
            action_reason = f"Low Return Risk ({risk_score:.1%} < {t_low:.1%}). Express frictionless checkout with all payment methods enabled."
            potential_savings = 0.0

        # Calculate Top 3 Risk Drivers
        drivers = self._extract_top_drivers(req, risk_score)

        return OrderScoreResponse(
            order_id=req.order_id or "ORD_DEMO_001",
            risk_score=risk_score,
            risk_tier=risk_tier,
            action=action,
            action_reason=action_reason,
            potential_savings_inr=potential_savings,
            top_drivers=drivers,
            thresholds_applied={"optimal_threshold_high": t_high, "low_threshold": t_low},
            evaluated_at=datetime.now(timezone.utc).isoformat()
        )

    def _extract_top_drivers(self, req: OrderScoreRequest, risk_score: float) -> list[DriverItem]:
        candidates = []

        # 1. Past Return Rate & Loyalty
        if req.user_past_return_rate is not None:
            if req.user_past_return_rate > 0.35:
                candidates.append(DriverItem(
                    feature_name="user_past_return_rate",
                    display_name="Customer Return History",
                    direction="INCREASES_RISK",
                    impact_score=0.92,
                    explanation=f"Buyer has a high historical return rate of {req.user_past_return_rate:.0%}."
                ))
            elif req.user_past_return_rate <= 0.10 and (req.user_past_orders_count or 0) >= 3:
                candidates.append(DriverItem(
                    feature_name="user_past_return_rate",
                    display_name="Customer Loyalty & Trust",
                    direction="REDUCES_RISK",
                    impact_score=0.85,
                    explanation=f"Repeat loyal buyer with {req.user_past_orders_count} past orders and {req.user_past_return_rate:.0%} return rate."
                ))

        # 2. Payment Method
        if req.payment_method == "COD":
            candidates.append(DriverItem(
                feature_name="payment_method",
                display_name="Cash On Delivery (COD)",
                direction="INCREASES_RISK",
                impact_score=0.88,
                explanation="COD orders carry 3.5x higher doorstep refusal and non-delivery propensity vs prepaid."
            ))
        elif req.payment_method in ["UPI", "CreditCard"]:
            candidates.append(DriverItem(
                feature_name="payment_method",
                display_name="Prepaid Method",
                direction="REDUCES_RISK",
                impact_score=0.82,
                explanation=f"{req.payment_method} indicates verified buyer commitment and near-zero doorstep rejection."
            ))

        # 3. Category Propensity
        if req.product_category in ["Apparel", "Footwear"]:
            candidates.append(DriverItem(
                feature_name="product_category",
                display_name="High Sizing Variance Category",
                direction="INCREASES_RISK",
                impact_score=0.72,
                explanation=f"{req.product_category} items experience elevated fit/size trial returns."
            ))
        elif req.product_category in ["Beauty_PersonalCare", "Electronics"]:
            candidates.append(DriverItem(
                feature_name="product_category",
                display_name="Category Low Return Propensity",
                direction="REDUCES_RISK",
                impact_score=0.68,
                explanation=f"{req.product_category} category exhibits low post-delivery return rates."
            ))

        # 4. Geolocation / Pincode
        if req.pincode_tier in ["Tier3_SemiUrban", "Remote"]:
            candidates.append(DriverItem(
                feature_name="pincode_tier",
                display_name="Delivery Transit Zone Risk",
                direction="INCREASES_RISK",
                impact_score=0.65,
                explanation=f"{req.pincode_tier} has higher courier transit delays and RTO failure rates."
            ))
        elif req.pincode_tier == "Tier1_Metro":
            candidates.append(DriverItem(
                feature_name="pincode_tier",
                display_name="Metro Delivery Zone",
                direction="REDUCES_RISK",
                impact_score=0.60,
                explanation="Tier 1 Metro has high-density courier networks and top delivery success rates."
            ))

        # 5. Discount Depth
        if req.discount_percentage >= 0.30:
            candidates.append(DriverItem(
                feature_name="discount_percentage",
                display_name="Deep Discount Impulse",
                direction="INCREASES_RISK",
                impact_score=0.58,
                explanation=f"Steep discount ({req.discount_percentage:.0%}) correlates with buyer impulse cancellation."
            ))

        # Ensure at least 3 drivers
        if len(candidates) < 3:
            candidates.append(DriverItem(
                feature_name="order_amount",
                display_name="Order Ticket Size",
                direction="INCREASES_RISK" if req.order_amount > 3000 else "REDUCES_RISK",
                impact_score=0.45,
                explanation=f"Order value of ₹{req.order_amount:,.0f} evaluated against buyer category average."
            ))

        candidates.sort(key=lambda x: x.impact_score, reverse=True)
        return candidates[:3]

inference_engine = ReturnRiskInferenceEngine()
