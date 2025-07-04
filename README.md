

```markdown
# 💳 Financial Fraud Detection API 🔍

A **Transformer-based Financial Fraud Detection system** with a FastAPI backend and Streamlit-powered frontend. Designed for detecting fraudulent transactions in real-time using advanced deep learning models.

---

## 🧠 Model Highlights

- ✅ **Transformer Architecture**
- 🎯 Trained with **Focal Loss** to handle class imbalance
- 🔄 Preprocessing with scaling and feature engineering
- 📊 Supports **real-time predictions** via API or UI

---

## 🔧 Project Structure

```

fraud\_detector/
│
├── backend/
│   ├── main.py                # FastAPI server
│   ├── model/
│   │   ├── focal\_loss.py
│   │   ├── scaler.pkl
│   │   └── transformer\_fraud\_model\_optimized.keras
│   └── requirements.txt
│
├── frontend/
│   ├── app.py                 # Streamlit app
│   └── requirements.txt
│
├── train\_model.py            # Model training script
├── scaler\_genration.py       # Scaler generation
├── request.py                # Sample test request (for FastAPI)
└── README.md

````

---

## 🚀 How to Run

### 📦 Backend (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
````

### 🎯 Streamlit Frontend

```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

---

## 📈 Prediction Endpoint

POST: `http://localhost:8000/predict/`

**Request Body:**

```json
{
  "features": [0.5, -1.2, 3.4, ..., 30 values total]
}
```

**Response:**

```json
{
  "fraud_probability": 0.9123,
  "prediction": "Fraud",
  "threshold": 0.9123
}
```

---

## 📚 Tech Stack

* 🧠 **TensorFlow / Keras**
* 🔁 **Transformer Architecture**
* ⚡ **FastAPI**
* 🌐 **Streamlit**
* 📊 **Scikit-learn**
* 🐍 **Python 3.10+**

---

## 👤 Author

**Sharukh S**
📫 [LinkedIn](https://www.linkedin.com/in/your-profile/) (update link)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

```

