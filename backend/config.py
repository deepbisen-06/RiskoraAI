import os
import json
from dataclasses import dataclass

@dataclass
class Settings:
    PROJECT_NAME: str = "Razorpay Return-Risk & RTO Prevention Engine"
    API_V1_STR: str = "/api/v1"
    DATABASE_URL: str = "sqlite:///./returns_audit.db"
    
    ARTIFACTS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ml_engine", "artifacts")
    MODEL_PATH: str = os.path.join(ARTIFACTS_DIR, "model.joblib")
    PREPROCESSOR_PATH: str = os.path.join(ARTIFACTS_DIR, "preprocessor.joblib")
    METRICS_PATH: str = os.path.join(ARTIFACTS_DIR, "metrics.json")
    
    # Financial Cost Parameters (INR)
    DEFAULT_C_FN: float = 350.0   # Reverse shipping + restocking loss per unintercepted RTO
    DEFAULT_C_FP: float = 70.0    # Gross margin friction loss if good buyer drops off
    UPI_CONVERSION_EFFICIENCY: float = 0.35 # Likelihood customer converts from COD to UPI on cashback nudge
    
    # Operational Thresholds (Synchronized dynamically with validation sweep)
    OPTIMAL_THRESHOLD_HIGH: float = 0.58
    OPTIMAL_THRESHOLD_LOW: float = 0.26
    
    def load_dynamic_thresholds(self):
        """Loads optimal T* derived from validation cost-weighted sweep in metrics.json"""
        if os.path.exists(self.METRICS_PATH):
            try:
                with open(self.METRICS_PATH, "r") as f:
                    data = json.load(f)
                    if "optimal_threshold" in data:
                        self.OPTIMAL_THRESHOLD_HIGH = float(data["optimal_threshold"])
                    if "low_threshold" in data:
                        self.OPTIMAL_THRESHOLD_LOW = float(data["low_threshold"])
                    print(f"[Config] Dynamically synchronized thresholds: HIGH(T*)={self.OPTIMAL_THRESHOLD_HIGH}, LOW={self.OPTIMAL_THRESHOLD_LOW}")
            except Exception as e:
                print(f"[Config] Warning: Could not parse metrics.json ({e}). Using defaults.")

settings = Settings()
settings.load_dynamic_thresholds()
