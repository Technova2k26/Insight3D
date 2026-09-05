"""
InSight3D - Prediction Pipeline Module
AI-Based Internal Defect Segmentation in 3D-Printed Electronic Components Using U-Net

Workflow:
1. Input CT Image
2. Preprocess (Resize, CLAHE, Noise Removal, Normalization)
3. Load Trained U-Net Model Checkpoint (`models/best_model.pth` or `.keras`)
4. Predict Probability Mask
5. Threshold Prediction -> Binary Mask
6. Save Binary Mask
7. Generate Color-Coded Overlay Image
"""

import os
import sys
import argparse
from typing import Tuple, Optional, Union
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocess import preprocess_pipeline, resize_image
from utils.metrics import get_custom_objects

try:
    import torch
    from model.unet_torch import UNetTorch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False


class DefectPredictor:
    """Defect Segmentation Predictor for CT Slice Images."""

    def __init__(
        self,
        model_path: Optional[str] = "models/best_model.pth",
        img_size: Tuple[int, int] = (256, 256),
        threshold: float = 0.5,
        device: str = "cpu"
    ):
        self.img_size = img_size
        self.threshold = threshold
        self.device = device
        self.model_type = "torch" if HAS_TORCH else "tf"
        self.model = self._load_or_create_model(model_path)

    def _load_or_create_model(self, model_path: Optional[str]):
        """Load trained model weights from disk or initialize baseline model."""
        # 1. Try PyTorch .pth checkpoint
        pth_path = model_path if (model_path and model_path.endswith(".pth")) else "models/best_model.pth"
        if HAS_TORCH and os.path.exists(pth_path):
            try:
                print(f"[INFO] Loading PyTorch U-Net checkpoint from: {pth_path}")
                checkpoint = torch.load(pth_path, map_location=self.device)
                init_filters = checkpoint.get("init_filters", 32)
                dropout_rate = checkpoint.get("dropout_rate", 0.1)

                model = UNetTorch(in_channels=1, out_channels=1, init_filters=init_filters, dropout_rate=dropout_rate).to(self.device)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                self.model_type = "torch"
                return model
            except Exception as e:
                print(f"[WARNING] Could not load PyTorch model from {pth_path}: {e}")

        # 2. Try TensorFlow/Keras .keras checkpoint
        keras_path = "models/best_model.keras"
        if HAS_TF and os.path.exists(keras_path):
            try:
                print(f"[INFO] Loading Keras U-Net model from: {keras_path}")
                model = tf.keras.models.load_model(keras_path, custom_objects=get_custom_objects())
                self.model_type = "tf"
                return model
            except Exception as e:
                print(f"[WARNING] Could not load Keras model from {keras_path}: {e}")

        # 3. Fallback baseline PyTorch model
        print("[INFO] Initializing baseline PyTorch U-Net architecture...")
        model = UNetTorch(in_channels=1, out_channels=1, init_filters=32).to(self.device)
        model.eval()
        self.model_type = "torch"
        return model

    def predict(
        self,
        image_input: Union[str, np.ndarray],
        use_clahe: bool = True,
        noise_reduction: Optional[str] = "gaussian",
        threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute prediction pipeline on input image path or array.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (preprocessed_img, prob_map, binary_mask)
        """
        thresh = threshold if threshold is not None else self.threshold

        if isinstance(image_input, str):
            raw_img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
            if raw_img is None:
                raise ValueError(f"Could not load CT image from path: {image_input}")
        elif isinstance(image_input, np.ndarray):
            raw_img = image_input.copy()
            if len(raw_img.shape) == 3 and raw_img.shape[2] == 3:
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        else:
            raise TypeError("image_input must be a file path string or numpy array.")

        prep_img = preprocess_pipeline(
            raw_img,
            target_size=self.img_size,
            is_mask=False,
            use_clahe=use_clahe,
            noise_reduction=noise_reduction,
            normalize=True
        )

        if self.model_type == "torch":
            tensor_inp = torch.tensor(prep_img[:, :, 0], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            with torch.no_grad():
                prob_map = self.model(tensor_inp).squeeze().cpu().numpy()
        else:
            input_tensor = np.expand_dims(prep_img, axis=0)
            prob_map = self.model.predict(input_tensor, verbose=0)[0, :, :, 0]

        binary_mask = (prob_map >= thresh).astype(np.uint8)

        return prep_img[:, :, 0], prob_map, binary_mask

    def create_overlay(
        self,
        original_img: np.ndarray,
        binary_mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 0, 255),
        alpha: float = 0.5
    ) -> np.ndarray:
        """Blend predicted defect binary mask onto original CT image."""
        if original_img.max() <= 1.0:
            base = (original_img * 255.0).astype(np.uint8)
        else:
            base = original_img.astype(np.uint8)

        if len(base.shape) == 2:
            base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        else:
            base_bgr = base.copy()

        color_mask = np.zeros_like(base_bgr)
        color_mask[binary_mask == 1] = color

        overlay = base_bgr.copy()
        mask_indices = binary_mask == 1
        overlay[mask_indices] = cv2.addWeighted(
            base_bgr[mask_indices], 1.0 - alpha, color_mask[mask_indices], alpha, 0
        )
        return overlay

    def process_and_save(
        self,
        image_input: Union[str, np.ndarray],
        output_mask_path: str = "results/masks/predicted_mask.png",
        output_overlay_path: str = "results/overlays/predicted_overlay.png"
    ) -> dict:
        """Full prediction, thresholding, and output file saving pipeline."""
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_overlay_path), exist_ok=True)

        prep_img, prob_map, binary_mask = self.predict(image_input)
        overlay = self.create_overlay(prep_img, binary_mask)

        cv2.imwrite(output_mask_path, binary_mask * 255)
        cv2.imwrite(output_overlay_path, overlay)

        return {
            "preprocessed_image": prep_img,
            "probability_map": prob_map,
            "binary_mask": binary_mask,
            "overlay_image": overlay,
            "saved_mask_path": output_mask_path,
            "saved_overlay_path": output_overlay_path
        }


if __name__ == "__main__":
    print("InSight3D Prediction Pipeline Test")
    test_ct = np.random.randint(80, 180, size=(256, 256), dtype=np.uint8)
    cv2.circle(test_ct, (128, 128), 25, 20, -1)

    predictor = DefectPredictor(model_path=None)
    results = predictor.process_and_save(
        test_ct,
        output_mask_path="results/masks/test_mask.png",
        output_overlay_path="results/overlays/test_overlay.png"
    )

    print(f"Prediction complete! Saved to {results['saved_mask_path']}")
