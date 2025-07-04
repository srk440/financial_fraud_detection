# Step 1: Imports
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Step 2: Define Base Path
base_path = r"D:\WORK\FRAUD DETECTION\Data_set"

# Step 3: Load Data
X_train = pd.read_csv(os.path.join(base_path, "X_train.xls"))
y_train = pd.read_csv(os.path.join(base_path, "y_train.xls"))
X_test  = pd.read_csv(os.path.join(base_path, "X_test.xls"))
y_test  = pd.read_csv(os.path.join(base_path, "y_test.xls"))

# If your files still have .xls extension but are really CSVs:
# Replace ".csv" with ".xls" above—pd.read_csv will still work.

# Step 4: Check for Missing Values
print("Missing in X_train:\n", X_train.isnull().sum())
print("Missing in y_train:\n", y_train.isnull().sum())

# Step 5: Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Handle Class Imbalance (SMOTE)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

# Step 7: Save Preprocessed Data Locally
out_dir = r"D:\WORK\FRAUD DETECTION\preprocessed_data"
os.makedirs(out_dir, exist_ok=True)

pd.DataFrame(X_train_resampled).to_csv(os.path.join(out_dir, "X_train_pre.csv"), index=False)
pd.DataFrame(y_train_resampled).to_csv(os.path.join(out_dir, "y_train_pre.csv"), index=False)
pd.DataFrame(X_test_scaled).to_csv(os.path.join(out_dir, "X_test_pre.csv"), index=False)
y_test.to_csv(os.path.join(out_dir, "y_test_pre.csv"), index=False)

print(" Data preprocessing is completed!")
print("Files saved to:", out_dir)
