import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Load training features
X_train = pd.read_csv("D:/WORK/FRAUD DETECTION/preprocessed_data/X_train.csv")

# Fit scaler on training data
scaler = StandardScaler()
scaler.fit(X_train)

# Save the fitted scaler
joblib.dump(scaler, "D:/WORK/FRAUD DETECTION/fraud_detector/backend/model/scaler.pkl")

print(" scaler.pkl generated and saved successfully.")
