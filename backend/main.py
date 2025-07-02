from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model("transformer_fraud_model.keras", compile=False)

class Transaction(BaseModel):
    features: list

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        data = np.array(transaction.features).reshape(1, 1, -1)
        prediction = model.predict(data)
        return {"fraud_probability": float(prediction[0][0])}
    except Exception as e:
        return {"error": str(e)}
