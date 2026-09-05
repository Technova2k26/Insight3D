"""
InSight3D - Metrics & Loss Functions Module
AI-Based Internal Defect Segmentation in 3D-Printed Electronic Components Using U-Net

Implements:
- Dice Coefficient
- Dice Loss
- IoU (Intersection over Union) / Jaccard Index
- Precision & Recall
- Combined BCE-Dice Loss
"""

import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    HAS_TF = True
except ImportError:
    HAS_TF = False
    tf = None


def dice_coef(y_true, y_pred, smooth: float = 1e-6):
    """Compute Dice Coefficient for binary segmentation."""
    if HAS_TF and isinstance(y_true, (tf.Tensor, tf.Variable)):
        y_true_f = K.flatten(K.cast(y_true, 'float32'))
        y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
        intersection = K.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    # NumPy fallback
    y_true_f = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred_f = np.asarray(y_pred, dtype=np.float32).ravel()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def dice_loss(y_true, y_pred, smooth: float = 1e-6):
    """Compute Dice Loss for U-Net optimization."""
    return 1.0 - dice_coef(y_true, y_pred, smooth=smooth)


def iou_score(y_true, y_pred, smooth: float = 1e-6):
    """Compute Intersection over Union (IoU / Jaccard Index)."""
    if HAS_TF and isinstance(y_true, (tf.Tensor, tf.Variable)):
        y_true_f = K.flatten(K.cast(y_true, 'float32'))
        y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
        intersection = K.sum(y_true_f * y_pred_f)
        total = K.sum(y_true_f) + K.sum(y_pred_f)
        union = total - intersection
        return (intersection + smooth) / (union + smooth)

    # NumPy fallback
    y_true_f = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred_f = np.asarray(y_pred, dtype=np.float32).ravel()
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def precision_m(y_true, y_pred, smooth: float = 1e-6):
    """Compute Precision metric."""
    if HAS_TF and isinstance(y_true, (tf.Tensor, tf.Variable)):
        y_true_f = K.flatten(K.cast(y_true, 'float32'))
        y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
        true_positives = K.sum(y_true_f * y_pred_f)
        predicted_positives = K.sum(y_pred_f)
        return (true_positives + smooth) / (predicted_positives + smooth)

    # NumPy fallback
    y_true_f = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred_f = np.asarray(y_pred, dtype=np.float32).ravel()
    true_positives = np.sum(y_true_f * y_pred_f)
    predicted_positives = np.sum(y_pred_f)
    return (true_positives + smooth) / (predicted_positives + smooth)


def recall_m(y_true, y_pred, smooth: float = 1e-6):
    """Compute Recall metric."""
    if HAS_TF and isinstance(y_true, (tf.Tensor, tf.Variable)):
        y_true_f = K.flatten(K.cast(y_true, 'float32'))
        y_pred_f = K.flatten(K.cast(y_pred, 'float32'))
        true_positives = K.sum(y_true_f * y_pred_f)
        possible_positives = K.sum(y_true_f)
        return (true_positives + smooth) / (possible_positives + smooth)

    # NumPy fallback
    y_true_f = np.asarray(y_true, dtype=np.float32).ravel()
    y_pred_f = np.asarray(y_pred, dtype=np.float32).ravel()
    true_positives = np.sum(y_true_f * y_pred_f)
    possible_positives = np.sum(y_true_f)
    return (true_positives + smooth) / (possible_positives + smooth)


def bce_dice_loss(y_true, y_pred):
    """Combined Binary Cross-Entropy and Dice Loss."""
    if HAS_TF and isinstance(y_true, (tf.Tensor, tf.Variable)):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        d_loss = dice_loss(y_true, y_pred)
        return K.mean(bce) + d_loss

    # NumPy fallback calculation
    y_t = np.clip(np.asarray(y_true, dtype=np.float32), 1e-7, 1 - 1e-7)
    y_p = np.clip(np.asarray(y_pred, dtype=np.float32), 1e-7, 1 - 1e-7)
    bce = - (y_t * np.log(y_p) + (1 - y_t) * np.log(1 - y_p))
    return np.mean(bce) + dice_loss(y_t, y_p)


def get_custom_objects() -> dict:
    """Return dictionary of custom metrics and loss functions for Keras model loading."""
    return {
        'dice_coef': dice_coef,
        'dice_loss': dice_loss,
        'iou_score': iou_score,
        'precision_m': precision_m,
        'recall_m': recall_m,
        'bce_dice_loss': bce_dice_loss
    }


if __name__ == "__main__":
    print("InSight3D Metrics & Loss Module Test")
    y_t = np.array([[[[0.0], [1.0]], [[1.0], [1.0]]]])
    y_p = np.array([[[[0.1], [0.9]], [[0.8], [0.9]]]])

    print(f"Dice Coefficient: {dice_coef(y_t, y_p):.4f}")
    print(f"Dice Loss:        {dice_loss(y_t, y_p):.4f}")
    print(f"IoU Score:        {iou_score(y_t, y_p):.4f}")
    print(f"Precision:        {precision_m(y_t, y_p):.4f}")
    print(f"Recall:           {recall_m(y_t, y_p):.4f}")
    print(f"BCE-Dice Loss:    {bce_dice_loss(y_t, y_p):.4f}")
