# 💳 Transformer-Based Financial Fraud Detection App

A full-stack fraud detection system using **deep learning Transformers** for real-time credit card fraud detection. Built with:
- ✅ TensorFlow (Keras)
- ✅ FastAPI (for backend APIs)
- ✅ Streamlit (for frontend dashboard)
- ✅ SMOTE (for handling class imbalance)
- ✅ Focal Loss (for better fraud classification)

---

## 🚀 Features

- Upload transaction data and get instant fraud predictions
- Real-time inference using a Transformer model trained on the credit card fraud dataset
- Interactive frontend (Streamlit) + scalable backend (FastAPI)
- Optimized for class imbalance with SMOTE + Focal Loss
- Supports `.keras`-based production deployment

---

## 📁 Project Structure

fraud_detector/
├── backend/ ← FastAPI backend with ML model loading
│ ├── main.py ← Handles API endpoints and model inference
│ ├── transformer_fraud_model.keras ← Trained Transformer model
│ └── requirements.txt
│
├── frontend/ ← Streamlit frontend for user interaction
│ ├── app.py ← Streamlit UI script
│ └── requirements.txt
│
├── model.py ← Model training script (Transformer)
├── 1_datac_collection_preprocess.py ← Preprocessing & data cleaning
├── README.md ← You're here!
└── .gitignore

---

## ⚙️ How to Run Locally

### 1️⃣ Start Backend (FastAPI)
```bash
cd fraud_detector/backend
uvicorn main:app --reload

2️⃣ Start Frontend (Streamlit)

cd fraud_detector/frontend
streamlit run app.py

🌐 Deploy on Streamlit Cloud
To deploy this app on Streamlit Community Cloud:

✅ Step 1: Push to GitHub
Push this folder to a public repository, e.g.:
https://github.com/srk440/fraud-detection-app



✅ Step 2: Deploy on Streamlit
Go to https://streamlit.io/cloud

Click “New App”

Connect your repo: srk440/fraud-detection-app

Set the app path to:
fraud_detector/frontend/app.py
Click Deploy

✅ Done!

🧠 Model Details
Dataset: Credit Card Fraud Detection (Kaggle)

Input Features: 30 anonymized transaction features

Model: Custom Transformer Encoder (3 layers)

Optimizer: Adam

Loss Function: Focal Loss (custom implementation)

Evaluation Metrics: ROC AUC, Precision-Recall, F1-Score

Data Handling: SMOTE for oversampling fraud class

📦 Python Package Requirements
📂 backend/requirements.txt

fastapi
uvicorn
tensorflow
pandas
numpy
scikit-learn
imblearn

📂 frontend/requirements.txt
streamlit
requests
pandas
numpy
🤝 Credits
Built with ❤️ by srk440

Transformer architecture adapted for tabular fraud detection

End-to-end deployment powered by Streamlit Cloud

📬 Contact
For collaborations or questions, feel free to connect:

GitHub: srk440

