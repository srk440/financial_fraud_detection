# frontend/app.py

import streamlit as st
import requests

st.set_page_config(page_title="Fraud Detection App", layout="wide")
st.title("💳 Real-Time Financial Fraud Detection")

st.markdown("Enter all 30 transaction features below:")

# Generate 30 input fields
features = []
cols = st.columns(3)
for i in range(30):
    with cols[i % 3]:
        val = st.number_input(f"Feature {i+1}", key=f"feature_{i+1}")
        features.append(val)

# On submit
if st.button("Predict"):
    with st.spinner("Sending data to backend..."):
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict/",
                json={"features": features}
            )
            if response.status_code == 200:
                result = response.json()
                st.success(f"🔍 Prediction: **{result['prediction']}**")
                st.metric("📊 Fraud Probability", f"{result['fraud_probability']:.4f}")
                st.caption(f"Threshold used: {result['threshold']:.4f}")
            else:
                st.error(f"Error {response.status_code}: {response.text}")
        except Exception as e:
            st.error(f"⚠️ Request failed: {e}")
