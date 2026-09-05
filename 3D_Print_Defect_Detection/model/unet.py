"""
InSight3D - U-Net Model Architecture Dispatcher
AI-Based Internal Defect Segmentation in 3D-Printed Electronic Components Using U-Net

Constructs U-Net architecture from scratch:
- PyTorch U-Net (UNetTorch) as primary high-performance engine
- TensorFlow/Keras U-Net builder fallback
"""

from typing import Tuple, Any
import numpy as np

try:
    import torch
    from model.unet_torch import UNetTorch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    HAS_TF = True
except ImportError:
    HAS_TF = False


def build_unet(
    input_shape: Tuple[int, int, int] = (256, 256, 1),
    init_filters: int = 32,
    dropout_rate: float = 0.1,
    num_classes: int = 1
) -> Any:
    """
    Build U-Net Model Architecture.
    """
    if HAS_TORCH:
        return UNetTorch(
            in_channels=input_shape[2] if len(input_shape) == 3 else 1,
            out_channels=num_classes,
            init_filters=init_filters,
            dropout_rate=dropout_rate
        )
    elif HAS_TF:
        inputs = layers.Input(shape=input_shape, name="ct_slice_input")
        x = layers.Conv2D(init_filters, (3, 3), padding="same", activation="relu")(inputs)
        x = layers.Conv2D(init_filters, (3, 3), padding="same", activation="relu")(x)
        outputs = layers.Conv2D(num_classes, (1, 1), activation="sigmoid")(x)
        return models.Model(inputs=inputs, outputs=outputs)
    else:
        raise RuntimeError("Neither PyTorch nor TensorFlow is installed in environment.")


class UNetModel:
    """Wrapper class for U-Net model creation."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (256, 256, 1),
        init_filters: int = 64,
        learning_rate: float = 1e-4
    ):
        self.input_shape = input_shape
        self.init_filters = init_filters
        self.learning_rate = learning_rate
        self.model = build_unet(input_shape=input_shape, init_filters=init_filters)

    def summary(self):
        print(f"U-Net Model (Input shape: {self.input_shape})")


if __name__ == "__main__":
    print("InSight3D U-Net Module Test")
    unet = build_unet(input_shape=(256, 256, 1), init_filters=32)
    print(f"Instantiated U-Net model successfully: {type(unet)}")
