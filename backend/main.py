import os
import json
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from backend.config import settings
from backend.database import engine, get_db, Base
from backend.models import OrderAuditLog
from backend.schemas import OrderScoreRequest, OrderScoreResponse, MetricsResponse
from backend.inference import inference_engine

# Initialize database schema
Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Reload thresholds and model artifacts
    settings.load_dynamic_thresholds()
    inference_engine.load_artifacts()
    yield

app = FastAPI(
    title="Riskora AI",
    description="AI-Powered Return & RTO Risk Intelligence for Checkout",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post(
    f"{settings.API_V1_STR}/score-order",
    response_model=OrderScoreResponse,
    summary="Score single checkout order for return/RTO risk"
)
def score_order(req: OrderScoreRequest, db: Session = Depends(get_db)):
    try:
        response = inference_engine.score_order(req)
        
        # Save audit log to SQLite
        log_record = OrderAuditLog(
            order_id=response.order_id,
            customer_id=req.customer_id,
            order_amount=req.order_amount,
            discount_percentage=req.discount_percentage,
            product_category=req.product_category,
            payment_method=req.payment_method,
            pincode_tier=req.pincode_tier,
            risk_score=response.risk_score,
            risk_tier=response.risk_tier.value,
            action=response.action.value,
            potential_savings_inr=response.potential_savings_inr,
            top_drivers_json=json.dumps([d.model_dump() for d in response.top_drivers])
        )
        db.add(log_record)
        db.commit()
        
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference scoring failed: {str(e)}"
        )

@app.get(
    f"{settings.API_V1_STR}/metrics",
    summary="Retrieve validation cost sweep, test set metrics, PR curves, and datacard"
)
def get_metrics():
    if not os.path.exists(settings.METRICS_PATH):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evaluation metrics not generated yet. Run python ml_engine/evaluate.py first."
        )
    try:
        with open(settings.METRICS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read metrics: {str(e)}"
        )

@app.get(
    f"{settings.API_V1_STR}/audit-logs",
    summary="Retrieve recent scored order audit records from database"
)
def get_audit_logs(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    try:
        logs = db.query(OrderAuditLog).order_by(desc(OrderAuditLog.created_at)).limit(limit).all()
        result = []
        for l in logs:
            drivers = []
            if l.top_drivers_json:
                try:
                    drivers = json.loads(l.top_drivers_json)
                except Exception:
                    drivers = []
            result.append({
                "id": l.id,
                "order_id": l.order_id,
                "customer_id": l.customer_id,
                "order_amount": l.order_amount,
                "discount_percentage": l.discount_percentage,
                "product_category": l.product_category,
                "payment_method": l.payment_method,
                "pincode_tier": l.pincode_tier,
                "risk_score": l.risk_score,
                "risk_tier": l.risk_tier,
                "action": l.action,
                "potential_savings_inr": l.potential_savings_inr,
                "top_drivers": drivers,
                "created_at": l.created_at.isoformat() if l.created_at else None
            })
        return {"total": len(result), "logs": result}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve audit logs: {str(e)}"
        )

@app.get(
    f"{settings.API_V1_STR}/dashboard-summary",
    summary="Retrieve live aggregate KPIs for overview dashboard"
)
def get_dashboard_summary(db: Session = Depends(get_db)):
    try:
        total_orders = db.query(func.count(OrderAuditLog.id)).scalar() or 0
        high_risk_count = db.query(func.count(OrderAuditLog.id)).filter(OrderAuditLog.risk_tier == "HIGH").scalar() or 0
        medium_risk_count = db.query(func.count(OrderAuditLog.id)).filter(OrderAuditLog.risk_tier == "MEDIUM").scalar() or 0
        low_risk_count = db.query(func.count(OrderAuditLog.id)).filter(OrderAuditLog.risk_tier == "LOW").scalar() or 0
        total_savings = db.query(func.sum(OrderAuditLog.potential_savings_inr)).scalar() or 0.0

        recent_logs = db.query(OrderAuditLog).order_by(desc(OrderAuditLog.created_at)).limit(6).all()
        recent = []
        for l in recent_logs:
            recent.append({
                "order_id": l.order_id,
                "customer_id": l.customer_id,
                "order_amount": l.order_amount,
                "payment_method": l.payment_method,
                "product_category": l.product_category,
                "risk_score": l.risk_score,
                "risk_tier": l.risk_tier,
                "action": l.action,
                "potential_savings_inr": l.potential_savings_inr,
                "created_at": l.created_at.isoformat() if l.created_at else None
            })

        return {
            "orders_scored": total_orders,
            "high_risk_orders": high_risk_count,
            "medium_risk_orders": medium_risk_count,
            "low_risk_orders": low_risk_count,
            "estimated_rtos_prevented": high_risk_count,
            "estimated_savings_inr": round(float(total_savings), 2),
            "recent_risk_checks": recent,
            "system_status": {
                "model": "XGBoost",
                "calibration": "Isotonic (CalibratedClassifierCV)",
                "explainability": "SHAP Feature Attribution",
                "api": "Healthy",
                "database": "Connected (SQLite)",
                "dataset": "Available (Synthetic + 10% Noise)"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load summary: {str(e)}"
        )

@app.get(f"{settings.API_V1_STR}/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Riskora AI",
        "model_loaded": inference_engine.model is not None,
        "database": "connected"
    }

# Mount frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

    @app.get("/", include_in_schema=False)
    def serve_frontend_root():
        index_path = os.path.join(frontend_dir, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Riskora AI running. Frontend index.html not found."}
