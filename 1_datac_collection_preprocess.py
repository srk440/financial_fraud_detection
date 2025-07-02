#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os

# Step 2: Load the Dataset (Fixed File Path)
file_path = r"D:\WORK\FRAUD DETECTION\DATA\creditcard.csv"  # Use raw string (r"")
df = pd.read_csv(file_path)

# Step 3: Check for Missing Values
print("Missing Values:\n", df.isnull().sum())

# Step 4: Data Overview
print(df.head())
print("Class Distribution Before SMOTE:\n", df["Class"].value_counts())

# Step 5: Normalize Transaction Amount
scaler = StandardScaler()
df["Amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))

# Step 6: Define Features & Target
X = df.drop(columns=["Class"])  # Features
y = df["Class"]  # Target (Fraud / Non-Fraud)

# Step 7: Split Data (Fixed Stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 8: Handle Imbalanced Data with SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Step 9: Check Class Balance After SMOTE
print("Class Distribution After SMOTE:\n", pd.Series(y_train_resampled).value_counts())

# Step 10: Save Preprocessed Data
os.makedirs("preprocessed_data", exist_ok=True)  # Create folder if not exists
X_train_resampled.to_csv("preprocessed_data/X_train.csv", index=False)
y_train_resampled.to_csv("preprocessed_data/y_train.csv", index=False)
X_test.to_csv("preprocessed_data/X_test.csv", index=False)
y_test.to_csv("preprocessed_data/y_test.csv", index=False)

print(" Data Preprocessing Complete! Ready for Model Training ")


# In[ ]:




