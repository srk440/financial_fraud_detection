
````markdown
# 💸 Financial Fraud Detection

**Transformer-based Financial Fraud Detection API with Streamlit Frontend**

This project demonstrates a robust, end-to-end pipeline for detecting financial fraud using deep learning. It includes a Transformer-based classification model, REST API via FastAPI, and an interactive dashboard built with Streamlit.

![Fraud Detection Banner](https://img.shields.io/badge/Made%20with-Transformers-00bcd4?style=for-the-badge) ![Status](https://img.shields.io/badge/Status-Production-green?style=for-the-badge)

---

## 📁 Dataset

- **Source:** [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Format:** `.xls` files (preprocessed and split)
- **Files Included:**
  - `X_train.xls`
  - `y_train.xls`
  - `X_test.xls`
  - `y_test.xls`

> *Note: The original dataset was converted to `.xls` format and split into training and test sets for convenience.*

---

## 🔄 Data Preprocessing

The following steps were applied in the `1_datac_collection_preprocess.py` and `scaler_generation.py` scripts:

1. **Loading raw `.xls` files** using `pandas`.
2. **Handling missing values** (if any).
3. **Feature scaling** using `StandardScaler`, saved as `scaler.pkl` for inference.
4. **Imbalanced data handling** using **SMOTE** to oversample minority (fraud) class.
5. **Train-test split** (80-20).
6. Saved preprocessed data as `.xls` files for model training.

---

## 🤖 Model Architecture

A **Transformer-based neural network** was built using `Keras` with the following components:

- `Input Layer`: Shape `(1, 30)`
- 3 Transformer blocks:
  - Multi-Head Self Attention
  - Residual Connections + Layer Normalization
  - Feed Forward Network
- `GlobalAveragePooling1D` → Dense Layers → Sigmoid Output

**Loss Function:** Focal Loss  
**Optimizer:** Adam  
**Evaluation Metric:** ROC-AUC + Precision/Recall  
**Output:** Binary classification (Fraud / Not Fraud)

### ✅ Model Summary:
- **Total Parameters:** `75,293`
- **Trainable Params:** `75,293`
- **Model Size:** ~294 KB

---

## 📈 Training Results

| Epoch | Accuracy | Val Accuracy | Loss     | Val Loss  |
|-------|----------|--------------|----------|-----------|
| 1     | 96.93%   | 98.62%       | 0.0011   | 1.88e-4   |
| 5     | 99.57%   | 99.82%       | 4.2e-5   | 2.27e-5   |
| 10    | 99.70%   | **99.82%**   | 2.6e-5   | **1.87e-5** |

### 📊 Evaluation (after threshold tuning at `0.894`):
- **ROC AUC Score:** `0.9232`
- **Classification Report**:
  - **Precision (Fraud):** 0.77
  - **Recall (Fraud):** 0.85
  - **F1 Score (Fraud):** 0.81
  - **Overall Accuracy:** 99.99%

---

## 🚀 Components

### 1. 🧠 Model Training
- File: `backend/model/model.py`
- Saves model as:
  - `transformer_fraud_model_optimized.h5`
  - `transformer_fraud_model_optimized.keras`

### 2. 🔌 Backend API (FastAPI)
- Endpoint: `POST /predict`
- Input: JSON with 30 feature values
- Output: Predicted class + Fraud probability

> Run with:
```bash
cd fraud_detector/backend
uvicorn main:app --reload
````

### 3. 🌐 Streamlit Frontend

* Path: `frontend/app.py`
* Collects user input, sends to FastAPI, and displays result

> Run with:

```bash
cd fraud_detector/frontend
streamlit run app.py
```

---

## 🗂 Directory Structure

```
fraud_detector/
├── backend/
│   ├── main.py
│   ├── model/
│   │   ├── model.py
│   │   ├── focal_loss.py
│   │   ├── transformer_fraud_model_optimized.h5
│   │   └── scaler.pkl
├── frontend/
│   └── app.py
├── Data_set/
│   ├── X_train.xls
│   ├── y_train.xls
│   ├── X_test.xls
│   └── y_test.xls
├── scaler_generation.py
├── train_model.py
├── request.py
└── README.md
```

---

## 🛠️ Setup Instructions

```bash
# Clone the repo
git clone https://github.com/srk440/financial_fraud_detection.git
cd financial_fraud_detection

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies
cd ../frontend
pip install -r requirements.txt
```

---

## 👨‍💻 Author

**Sharukh S**
🔗 [LinkedIn Profile](https://www.linkedin.com/in/sharukh-s-4992b325a/)

---

## 📌 License

This project is open-source and available under the [MIT License](LICENSE).

---

## ⭐ If you found this helpful, give the repo a star!

```

---

Let me know if you'd like me to auto-create and upload this as a file (`README.md`) in your GitHub repo.
```
