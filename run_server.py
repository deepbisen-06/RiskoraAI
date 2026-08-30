import os
import sys
import uvicorn

def main():
    print("=" * 70)
    print("   RAZORPAY RETURN-RISK SCORER & RTO PREVENTION ENGINE")
    print("=" * 70)
    
    # Check if trained model artifacts exist
    artifacts_dir = os.path.join(os.path.dirname(__file__), "ml_engine", "artifacts")
    model_path = os.path.join(artifacts_dir, "model.joblib")
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    
    if not os.path.exists(model_path) or not os.path.exists(metrics_path):
        print("\n[Setup] Model artifacts not found. Initiating training & evaluation pipeline...")
        from ml_engine.train import train_models
        from ml_engine.evaluate import evaluate_models
        
        train_models()
        evaluate_models()
        print("\n[Setup] Training and evaluation complete.")
    else:
        print("\n[Setup] Found existing calibrated model artifacts and evaluation metrics.")
        
    print("\n[Server] Starting FastAPI backend and demo web surface on http://127.0.0.1:8000 ...")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
