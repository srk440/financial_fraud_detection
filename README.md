```markdown
# 💳 Financial Fraud Detection

Transformer-based Financial Fraud Detection API with a lightweight Streamlit frontend.

![GitHub Repo](https://img.shields.io/badge/Status-Production--Ready-green?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-sharukh--s-blue?logo=linkedin&style=flat-square)](https://www.linkedin.com/in/sharukh-s-4992b325a/)

---

## 🧠 Overview

This project leverages a custom **Transformer-based neural network** to detect **financial fraud** from transactional data. It is equipped with:

- A deep transformer model with attention layers
- Focal Loss for imbalance handling
- Precision-Recall threshold optimization
- Streamlit interface for real-time fraud prediction

---

## 🚀 Features

- Transformer encoder stack with residuals and normalization
- Trained with **class imbalance** using **Focal Loss**
- Real-time API with FastAPI backend
- Simple UI using Streamlit frontend
- Threshold-tuned binary classification for fraud detection

---

## 📁 Folder Structure

```

financial\_fraud\_detection/
├── backend/
│   ├── main.py
│   ├── model/
│   │   ├── transformer\_fraud\_model\_optimized.keras
│   │   ├── scaler.pkl
│   │   ├── focal\_loss.py
├── frontend/
│   ├── app.py
├── preprocessed\_data/
├── Data\_set/
├── scaler\_genration.py
├── train\_model.py
├── request.py
├── 1\_datac\_collection\_preprocess.py
├── README.md

````

---

## 🛠️ Tech Stack

| Component     | Tool/Library     |
|---------------|------------------|
| Language      | Python 3.10+     |
| ML Framework  | TensorFlow/Keras |
| Backend API   | FastAPI          |
| Frontend UI   | Streamlit        |
| Data Handling | Pandas, NumPy    |

---

## 🔬 Model Architecture

- **Input**: (None, 1, 30)
- **Transformer Layers**: 3 blocks with MultiHeadAttention, Add & Norm, FeedForward
- **Pooling**: GlobalAveragePooling1D
- **Final Dense Layers**: [64, 32, 1]
- **Loss**: Focal Loss (γ = 2.0)
- **Optimizer**: Adam
- **Epochs**: 10

```text
Total Trainable Parameters: 75,293 (~294 KB)
````

---

## 📈 Model Performance

### ✅ Training Results

| Epoch | Accuracy | Loss      | Val Accuracy | Val Loss  |
| ----- | -------- | --------- | ------------ | --------- |
| 1     | 96.93%   | 0.0011    | 98.62%       | 0.0001877 |
| 5     | 99.57%   | 0.0000421 | 99.82%       | 0.0000227 |
| 10    | 99.70%   | 0.0000263 | 99.82%       | 0.0000187 |

> ✅ Optimized with Best Threshold: `0.8941`

### 🔍 Evaluation (Test Set)

| Metric            | Value      |
| ----------------- | ---------- |
| Accuracy          | 99.99%     |
| Precision (Fraud) | 77%        |
| Recall (Fraud)    | 85%        |
| F1 Score (Fraud)  | 81%        |
| ROC AUC Score     | **0.9232** |

> 🧠 Fraud samples: 98 | Non-Fraud: 56,864

---

## 📦 Installation

```bash
# Backend
cd backend/
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd ../frontend/
streamlit run app.py
```

---

## 📡 API Usage

**POST** `/predict/`

**Request JSON:**

```json
{
  "features": [0.1, 0.2, ..., 0.05]  // 30 float features
}
```

**Response JSON:**

```json
{
  "prediction": 0,
  "fraud_probability": 0.0517
}
```

---

## 👨‍💻 Author

**Sharukh S.**

* 🎓 B.Tech Artificial Intelligence & Data Science
* 💼 Intern @ University Digital Marketing Team
* 🔗 [LinkedIn Profile](https://www.linkedin.com/in/sharukh-s-4992b325a/)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

```

---

Let me know if you'd like:
- A `LICENSE` file (MIT/GPL/Apache?)
- `.gitignore` optimized for Python + Streamlit
- Deployment instructions (e.g. Hugging Face Spaces, Render)
- Badges like CodeCov, Build Status, etc.

Ready to push this to GitHub?
```
