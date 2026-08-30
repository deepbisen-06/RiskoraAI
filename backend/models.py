from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from backend.database import Base

class OrderAuditLog(Base):
    __tablename__ = "order_audit_logs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    order_id = Column(String(64), index=True, nullable=False)
    customer_id = Column(String(64), index=True, nullable=False)
    order_amount = Column(Float, nullable=False)
    discount_percentage = Column(Float, nullable=False)
    product_category = Column(String(64), nullable=False)
    payment_method = Column(String(32), nullable=False)
    pincode_tier = Column(String(32), nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_tier = Column(String(16), nullable=False)
    action = Column(String(32), nullable=False)
    potential_savings_inr = Column(Float, nullable=False)
    top_drivers_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
