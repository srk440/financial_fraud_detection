# backend/main.py

import os
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from keras.models import load_model
from keras.utils import get_custom_objects
import joblib

# Correct import based on updated package structure
from .model.focal_loss import focal_loss_fixed


# Register custom loss function
get_custom_objects().update({"focal_loss_fixed": focal_loss_fixed})

# Define paths relative to current file
BASE_PATH = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(BASE_PATH, "transformer_fraud_model_optimized.keras")
SCALER_PATH = os.path.join(BASE_PATH, "scaler.pkl")

# Load the trained model and scaler
try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

try:
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading scaler: {e}")

# Initialize FastAPI app
app = FastAPI(title="Financial Fraud Detection API")

# Define input schema
class TransactionData(BaseModel):
    features: list[float]

# Prediction endpoint
@app.post("/predict/")
def predict(data: TransactionData):
    try:
        # Prepare input
        input_array = np.array(data.features).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        transformed_input = np.expand_dims(scaled_input, axis=1)

        # Predict probability
        probability = model.predict(transformed_input)[0][0]
        threshold = 0.9123
        prediction = int(probability > threshold)

        # Return response
        return {
            "fraud_probability": float(probability),
            "prediction": "Fraud" if prediction == 1 else "Legit",
            "threshold": threshold
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
