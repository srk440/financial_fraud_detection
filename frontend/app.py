import streamlit as st
import requests

st.title("💳 Financial Fraud Detection")

st.write("Enter transaction features (length: 30):")
features = [st.number_input(f"Feature {i+1}", value=0.0) for i in range(30)]

if st.button("Predict Fraud"):
    payload = {"features": features}
    res = requests.post("http://localhost:8000/predict", json=payload)
    if res.status_code == 200:
        prob = res.json()["fraud_probability"]
        st.success(f"🚨 Fraud Probability: {prob:.4f}")
    else:
        st.error("Error in prediction request.")
