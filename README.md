# Riskora AI — Return & RTO Risk Intelligence

<div align="center">

![Riskora AI Hero](docs/images/01_overview.png)

**AI-Powered Return & RTO (Return to Origin) Risk Intelligence for Checkout**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-Calibrated-orange.svg)](https://xgboost.readthedocs.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E.svg?logo=scikit-learn)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Build-Passing-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🌟 Executive Summary

**Riskora AI** is a machine learning system and real-time decisioning engine designed for e-commerce checkout integration. It predicts return / RTO (Return to Origin) propensity at the exact moment of order placement and executes **dynamic payment gating** (allowing COD, incentivizing UPI with instant cashback, or enforcing prepaid-only) to safeguard merchant margin.

### The Core Problem
- **High Reverse Logistics Loss**: In Indian and global e-commerce, Cash on Delivery (COD) orders suffer an average return/RTO rate of 25%–40%, costing merchants ₹250–₹400 in reverse shipping, damaged packaging, and locked inventory.
- **Blunt Blanket Bans Cause Churn**: Blanket blocking of COD alienates legitimate customers, causing severe top-line revenue drop-off.
- **The Riskora Solution**: Fine-grained, calibrated risk estimation coupled with cost-weighted threshold optimization ($T^*$) that intercepts high-risk doorstep refusals while preserving seamless checkouts for trusted buyers.

---

## 🏗️ System Workflow & Architecture

```mermaid
flowchart TD
    subgraph 1. ML Engine [ml_engine/]
        DATA[Sequential Order Stream<br/>60k Orders + 10% Label Noise] --> FE[Expanding Window Features<br/>Strict Temporal Shift(1)]
        FE --> SPLIT[Chronological Split<br/>Train: 70% | Val: 15% | Held-Out Test: 15%]
        SPLIT --> BASE[Baseline: Logistic Regression]
        SPLIT --> XGB[Trained XGBoost Classifier]
        XGB --> CAL[Probability Calibration<br/>Isotonic on Val Split]
        CAL --> TUNE[Cost-Weighted Threshold Sweep<br/>C_fn vs C_fp &rarr; Optimal T*]
        TUNE --> EVAL[Single-Pass Evaluation on Test Split<br/>PR-AUC: 0.638 vs Baseline: 0.584]
    end

    subgraph 2. FastAPI Decisioning Service [backend/]
        EVAL --> ART[(artifacts/<br/>model.joblib, metrics.json)]
        ART --> CONF[config.py<br/>Dynamically Loads T* = 0.32]
        CONF --> INF[inference.py<br/>Calibrated Probability + SHAP Drivers]
        API[FastAPI Endpoints<br/>POST /score-order | GET /metrics | GET /audit-logs] --> INF
        INF --> DB[(SQLite: returns_audit.db<br/>Tamper-Evident Audit Log)]
    end

    subgraph 3. Modern Fintech Command Center [frontend/]
        API --> V1[Page 1: Overview Command Center]
        API --> V2[Page 2: Live Checkout Simulator & Payment Gate]
        API --> V3[Page 3: Model Performance Analytics]
        API --> V4[Page 4: Threshold & ROI Studio]
        API --> V5[Page 5: Prediction Audit Trail]
        API --> V6[Page 6: Settings & Dataset Transparency Card]
    end
```

---

## 🔬 Methodological Rigor & Formulations

### 1. Leakage-Proof Temporal Feature Engineering
Customer return frequency, order count, and delivery zone risk are computed exclusively using historical data prior to the current order timestamp:
$$\text{user\_past\_return\_rate}_t = \frac{\sum_{i < t} \text{is\_returned}_i + 1}{\sum_{i < t} \mathbf{1}_i + 5}$$
*(Unit test verified in `ml_engine/tests/test_leakage.py` — asserts zero dependency on future rows).*

### 2. Validation Cost-Weighted Threshold Optimization ($T^*$)
The decision cutoff $T^*$ is selected on the **validation split only** by sweeping $T \in [0.15, 0.85]$ against business unit economics:
$$\text{Profit Saved}(T) = C_{FN} \cdot TP(T) - C_{FP} \cdot FP(T)$$
where:
- $C_{FN} = ₹350$: Reverse logistics and damaged stock loss per unintercepted RTO.
- $C_{FP} = ₹175$: Lost gross product margin if a good customer drops off due to disabled COD.
- $T^* = \arg\max_T \text{Profit Saved}(T) = 0.32$ (with derived $T_{\text{low}} = 0.14$).

### 3. Dynamic Payment Gating Matrix & Deterministic Savings
```
Probability p >= T* (0.32)       --> HIGH RISK   --> BLOCK_COD_REQUIRE_PREPAID
T_low <= p < T* (0.14 - 0.32)   --> MEDIUM RISK --> NUDGE_UPI_CASHBACK (₹50 discount)
p < T_low (< 0.14)              --> LOW RISK    --> ALLOW_COD (Express Checkout)
```

**Deterministic INR Savings Formula**:
$$\text{potential\_savings\_inr} = \begin{cases} C_{FN} \times p_{\text{return}} & \text{if action is } \texttt{BLOCK\_COD\_REQUIRE\_PREPAID} \\ C_{FN} \times p_{\text{return}} \times 0.35 & \text{if action is } \texttt{NUDGE\_UPI\_CASHBACK} \\ 0 & \text{if action is } \texttt{ALLOW\_COD} \end{cases}$$

---

## 📸 Visual Showcase & Feature Suite

### 1. Executive Command Center (Overview)
Real-time KPI cards, live risk distribution donut chart, system health indicators, and AI Assistant.
![Overview](docs/images/01_overview.png)

### 2. Live Checkout Simulator (High Risk & SHAP Attributions)
Instant checkout simulation showing dynamic COD blocking, calibrated return probability, and top 3 SHAP drivers.
![Live Scoring High Risk](docs/images/02_live_scoring_high_risk.png)

### 3. Live Scoring (Medium Risk with UPI Cashback Nudge)
Borderline risk scenario where COD remains enabled but an animated ₹50 UPI discount banner incentivizes conversion to prepaid.
![Live Scoring Medium Risk](docs/images/03_live_scoring_medium_risk.png)

### 4. Live Scoring (Low Risk Prime VIP)
Established loyal buyer with seamless express checkout and zero friction.
![Live Scoring Low Risk](docs/images/04_live_scoring_low_risk.png)

---

## 📂 Project Repository Structure

```
RiskoraAI/
├── ml_engine/
│   ├── generate_dataset.py      # E-commerce dataset generator + 10% label noise
│   ├── features.py              # Time-safe expanding window aggregates & pipeline
│   ├── train.py                 # Baseline Logistic Regression + XGBoost + Calibration
│   ├── evaluate.py              # Validation threshold sweep (T*) + test metrics
│   ├── tests/
│   │   ├── test_leakage.py      # Unit tests asserting zero future data leakage
│   │   └── test_pipeline.py     # Pipeline & feature transformation tests
│   └── artifacts/               # model.joblib, preprocessor.joblib, metrics.json
├── backend/
│   ├── main.py                  # FastAPI service (POST /score-order, GET /metrics, GET /audit-logs)
│   ├── config.py                # Configuration & dynamic T* threshold loader
│   ├── schemas.py               # Pydantic v2 schemas for request, response, and enums
│   ├── database.py              # SQLite connection & session management
│   ├── models.py                # SQLAlchemy OrderAuditLog database model
│   ├── inference.py             # Model inference engine & SHAP top driver extractor
│   └── tests/
│       └── test_api.py          # API integration tests
├── frontend/
│   ├── index.html               # 6-View Light Fintech Command Center UI
│   ├── app.js                   # Reactive state, Chart.js integrations, presets, ROI sliders
│   └── styles.css               # Clean fintech styling, animations, and risk tier badges
├── docs/
│   └── images/                  # High-resolution screenshots
├── run_server.py                # Unified one-command startup script
├── requirements.txt             # Pinned dependencies
├── .gitignore                   # Git ignore specifications
└── README.md                    # System documentation & technical spec
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Installation
```bash
# Clone repository
git clone https://github.com/deepbisen-06/RiskoraAI.git
cd RiskoraAI

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Automated Tests
```bash
python -m pytest ml_engine/tests/ backend/tests/ -v
```
*Expected output: 6 passed (100% test coverage for leakage, preprocessors, and API scoring).*

### 4. Train & Launch Unified Server
```bash
python run_server.py
```

- **Web Dashboard**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs**: Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🔌 API Reference

### `POST /api/v1/score-order`
Scores a single checkout transaction and executes dynamic payment gating.

**Request Payload:**
```json
{
  "order_id": "ORD_789412",
  "customer_id": "CUST_008412",
  "order_amount": 2499.0,
  "discount_percentage": 0.45,
  "product_category": "Apparel",
  "item_quantity": 2,
  "payment_method": "COD",
  "pincode_tier": "Remote",
  "shipping_speed": "Standard",
  "device_type": "Mobile_App",
  "user_past_orders_count": 4,
  "user_past_return_rate": 0.80
}
```

**Response Payload:**
```json
{
  "order_id": "ORD_789412",
  "risk_score": 0.7482,
  "risk_tier": "HIGH",
  "action": "BLOCK_COD_REQUIRE_PREPAID",
  "action_reason": "High RTO Risk (74.8% >= T*=32.0%). COD payment disabled to protect reverse logistics cost.",
  "potential_savings_inr": 261.87,
  "top_drivers": [
    {
      "feature_name": "user_past_return_rate",
      "display_name": "Customer Return History",
      "direction": "INCREASES_RISK",
      "impact_score": 0.92,
      "explanation": "Buyer has a high historical return rate of 80%."
    },
    {
      "feature_name": "payment_method",
      "display_name": "Cash On Delivery (COD)",
      "direction": "INCREASES_RISK",
      "impact_score": 0.88,
      "explanation": "COD orders carry 3.5x higher doorstep refusal and non-delivery propensity vs prepaid."
    },
    {
      "feature_name": "product_category",
      "display_name": "High Sizing Variance Category",
      "direction": "INCREASES_RISK",
      "impact_score": 0.72,
      "explanation": "Apparel items experience elevated fit/size trial returns."
    }
  ],
  "thresholds_applied": {
    "optimal_threshold_high": 0.32,
    "low_threshold": 0.14
  },
  "evaluated_at": "2026-08-30T10:45:00.000000Z"
}
```

---

## 📊 Evaluation Benchmark

| Metric | Baseline (Logistic Regression) | Calibrated XGBoost | Empirical Lift (Δ) |
| :--- | :--- | :--- | :--- |
| **PR-AUC** | 0.5842 | **0.6384** | **+0.0542 (+9.3%)** |
| **ROC-AUC** | 0.6211 | **0.6580** | **+0.0369** |
| **Precision @ T\*** | 0.4224 | **0.4682** | **+0.0458** |
| **Recall @ T\*** | 0.6691 | **0.6389** | Balanced |
| **Net Profit Saved (Test)** | ₹2,38,000 | **₹3,67,500** | **+₹1,29,500 (+54.4%)** |

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
