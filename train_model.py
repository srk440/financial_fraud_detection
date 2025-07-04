# model_train.py

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
import tensorflow.keras.backend as K
from tensorflow.keras.losses import BinaryCrossentropy
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve

#  Serializable Focal Loss Function
def focal_loss_fixed(y_true, y_pred, gamma=2.0, alpha=0.25):
    bce = BinaryCrossentropy()(y_true, y_pred)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    return K.mean(alpha * K.pow(1 - p_t, gamma) * bce)

#  Transformer Encoder Block
def transformer_encoder(inputs, head_size=64, num_heads=2, ff_dim=128, dropout=0.3):
    attention = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    attention = Dropout(dropout)(attention)
    attention = LayerNormalization(epsilon=1e-6)(attention + inputs)
    ff_output = Dense(ff_dim, activation="relu")(attention)
    ff_output = Dropout(dropout)(ff_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    return LayerNormalization(epsilon=1e-6)(ff_output + attention)

#  Build Transformer Model
def build_transformer_model(input_shape):
    inputs = Input(shape=input_shape)
    x = transformer_encoder(inputs)
    x = transformer_encoder(x)
    x = transformer_encoder(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=focal_loss_fixed,
        metrics=["accuracy"]
    )
    return model

#  Load Data
base_path = r"D:\WORK\FRAUD DETECTION\preprocessed_data"
X_train = pd.read_csv(os.path.join(base_path, "X_train_pre.csv")).values
y_train = pd.read_csv(os.path.join(base_path, "y_train_pre.csv")).values
X_test = pd.read_csv(os.path.join(base_path, "X_test_pre.csv")).values
y_test = pd.read_csv(os.path.join(base_path, "y_test_pre.csv")).values

#  Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#  Oversampling
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

#  Reshape for Transformer
X_train_resampled = np.expand_dims(X_train_resampled, axis=1)
X_test = np.expand_dims(X_test, axis=1)

#  Build & Train
input_shape = (X_train_resampled.shape[1], X_train_resampled.shape[2])
model = build_transformer_model(input_shape)
model.summary()

#  Class Weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_resampled),
    y=y_train_resampled
)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

#  Train
history = model.fit(
    X_train_resampled,
    y_train_resampled,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test),
    class_weight=class_weight_dict
)

#  Find Best Threshold
y_prob = model.predict(X_test)
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
best_threshold = thresholds[np.argmax(precisions * recalls)]
print(f"Best Threshold: {best_threshold:.4f}")

#  Evaluate
y_pred_new = (y_prob > best_threshold).astype("int32")
print("Updated Classification Report:\n", classification_report(y_test, y_pred_new))
print("Updated ROC AUC Score:", roc_auc_score(y_test, y_pred_new))

#  Save Model in All Compatible Formats
model_dir = r"D:\WORK\FRAUD DETECTION\fraud_detector\backend\model"
os.makedirs(model_dir, exist_ok=True)

# 1. Save as HDF5 (.h5) – for Keras 3 compatibility
model_h5_path = os.path.join(model_dir, "transformer_fraud_model_optimized.h5")
model.save(model_h5_path)
print(f" Model saved in HDF5 format at:\n{model_h5_path}")

# 2. Save as .keras – preferred for Keras 3+ future-proofing
model_keras_path = os.path.join(model_dir, "transformer_fraud_model_optimized.keras")
model.save(model_keras_path)
print(f" Model saved in KERAS format at:\n{model_keras_path}")

