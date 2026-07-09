import os
import joblib
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from app.schemas import TransactionPayload

# 1. Define global storage paths and variables
# Dynamically locate files relative to main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "fraud_scaler.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "fraud_lightgbm_model.pkl")
OPTIMIZED_THRESHOLD = 0.80

scaler = None
model = None

# 2. Define the Lifespan Context Manager to handle startup cleanly
@asynccontextmanager
async def lifespan(app: FastAPI):
    global scaler, model
    # Everything here runs ON STARTUP
    if not os.path.exists(SCALER_PATH) or not os.path.exists(MODEL_PATH):
        raise RuntimeError("Missing model deployment artifacts. Ensure .pkl files are in the directory.")
    
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)
    print("=== Production ML Artifacts Successfully Loaded (Modern Lifespan) ===")
    
    yield  # The server stays active here while serving requests
    
    # Anything written here runs ON SHUTDOWN
    print("=== Shutting Down Server ===")

# 3. Initialize the FastAPI Application passing the lifespan setup
app = FastAPI(
    title="Enterprise Credit Card Fraud Detection Microservice",
    description="A high-performance LightGBM-backed microservice for real-time transaction scoring.",
    version="1.0.0",
    lifespan=lifespan
)

# 4. Define the strict data validation schema for incoming API requests
class TransactionPayload(BaseModel):
    # Time and Amount require scaling; V1-V28 are pre-transformed components
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    Amount: float = Field(..., description="Transaction transaction amount in local currency")
    V_features: list[float] = Field(..., min_items=28, max_items=28, description="The 28 PCA-transformed feature columns (V1 to V28)")

# 5. Define the health check endpoint for monitoring systems (Docker/Kubernetes)
@app.get("/health", status_code=200)
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

# 6. Define the real-time scoring engine endpoint
@app.post("/v1/predict", status_code=200)
def predict_fraud(payload: TransactionPayload):
    try:
        # Extract features from the validated JSON payload
        time_val = payload.Time
        amount_val = payload.Amount
        v_features = payload.V_features
        
        # Format unscaled items to match the training pipeline's scaler shapes
        unscaled_inputs = np.array([[time_val, amount_val]])
        scaled_inputs = scaler.transform(unscaled_inputs)  # Inline transform rule
        
        # Assemble the full 30-feature vector in the precise training index order:
        # [Scaled_Time, V1, V2, ..., V28, Scaled_Amount]
        full_feature_vector = np.zeros((1, 30))
        full_feature_vector[0, 0] = scaled_inputs[0, 0]    # Scaled Time
        full_feature_vector[0, 1:29] = v_features           # V1 to V28
        full_feature_vector[0, 29] = scaled_inputs[0, 1]   # Scaled Amount
        
        # Generate the raw probability score
        fraud_probability = float(model.predict_proba(full_feature_vector)[0, 1])
        
        # Apply the calibrated operational threshold (0.80) to maximize precision
        prediction_verdict = 1 if fraud_probability >= OPTIMIZED_THRESHOLD else 0
        
        return {
            "is_fraud": prediction_verdict,
            "fraud_probability": round(fraud_probability, 4),
            "applied_threshold": OPTIMIZED_THRESHOLD,
            "action_required": "HOLD_TRANSACTION_BLOCK_CARD" if prediction_verdict == 1 else "ALLOW_TRANSACTION"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Pipeline Error: {str(e)}")