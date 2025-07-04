# backend/model/focal_loss.py

import tensorflow.keras.backend as K
from tensorflow.keras.losses import BinaryCrossentropy

def focal_loss_fixed(y_true, y_pred, gamma=2.0, alpha=0.25):
    bce = BinaryCrossentropy()(y_true, y_pred)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    return K.mean(alpha * K.pow(1 - p_t, gamma) * bce)
