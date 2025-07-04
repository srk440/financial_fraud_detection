 💸 Financial Fraud Detection using Transformers

This project is a full-stack **AI-powered Fraud Detection System** that uses Transformer-based deep learning models to identify fraudulent financial transactions in real time. The backend is built using **FastAPI**, and a lightweight UI is available via **Streamlit**.

---

## 📁 Project Structure

fraud_detector/
├── backend/
│ ├── main.py # FastAPI backend API
│ ├── requirements.txt # Backend dependencies
│ ├── model/
│ │ ├── transformer_fraud_model_optimized.keras
│ │ ├── scaler.pkl
│ │ └── focal_loss.py
├── frontend/
│ ├── app.py # Streamlit frontend UI
│ ├── requirements.txt
├── preprocessed_data/ # Processed training data (CSV)
├── 1_datac_collection_preprocess.py
├── train_model.py # Training script for Transformer
├── scaler_genration.py
├── request.py # Testing API endpoint using requests
└── README.md # This file

---

## 🔍 Features

- ✅ Transformer-based binary classifier (Fraud / Legit)
- ✅ Real-time fraud prediction API using **FastAPI**
- ✅ Frontend interface with **Streamlit**
- ✅ Custom **Focal Loss** for imbalanced datasets
- ✅ Scaler saved for consistent input preprocessing

---

## 🧠 Model Overview

- **Input**: 30 numerical features from financial transaction records  
- **Output**: Probability score & Fraud/Legit classification  
- **Model Used**: Transformer with custom architecture  
- **Loss Function**: Focal Loss (to handle class imbalance)

---

## 🚀 How to Run

### 1️⃣ Backend API (FastAPI)
```bash
cd fraud_detector
uvicorn backend.main:app --reload

###2️⃣ Frontend App (Streamlit)
```bash
cd fraud_detector/frontend
streamlit run app.py

📦 Dependencies
Backend:
	fastapi
	uvicorn
	tensorflow / keras
	joblib
	numpy
Frontend:
	streamlit
	http
Install them via:
	pip install -r backend/requirements.txt
	pip install -r frontend/requirements.txt
API Endpoint
	URL: POST /predict/

	Payload:
		{
		  "features": [0.1, -0.2, ..., 1.5]  // 30 float values
		}

	Response:
		{
		  "prediction": "Fraud",
		  "fraud_probability": 0.9421,
		  "threshold": 0.9123
		}
Model Details:
	Input: 30 continuous features

	Architecture: Custom Transformer block with dense layers

	Loss Function: Focal Loss (for imbalance handling)

	Output: Binary probability of fraud
Status:
✅ Model training complete
✅ Real-time backend running
✅ UI working with 30 feature inputs
🔜 Flutter app integration (optional, paused)
🔜 Deployment to Hugging Face / Render

 Author
Sharukh S
AI & Data Science Enthusiast | Placement-focused 


---

Let me know when you're ready to push this to GitHub — I can guide you step-by-step to commit, create the repo, push, and even publish on Hugging Face or Render!


