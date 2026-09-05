"""
InSight3D - Data Augmentation Module
AI-Based Internal Defect Segmentation in 3D-Printed Electronic Components Using U-Net

Functions for simultaneous image and mask augmentation:
- Horizontal & Vertical Flips
- Rotation (-30 to +30 degrees)
- Brightness & Contrast Adjustment
- Random Zooming
- Synchronized transformations preserving spatial alignment between CT images and ground truth defect masks.
"""

import random
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np


class DefectDataAugmentor:
    """
    Data Augmentor for synchronized CT slice images and defect segmentation masks.
    """

    def __init__(
        self,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        rotation_range: float = 30.0,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        zoom_range: Tuple[float, float] = (0.85, 1.15),
        p: float = 0.5
    ):
        """
        Args:
            horizontal_flip (bool): Enable random horizontal flip.
            vertical_flip (bool): Enable random vertical flip.
            rotation_range (float): Maximum degree range for random rotation [-deg, +deg].
            brightness_range (Tuple[float, float]): Factor range for brightness adjustment.
            contrast_range (Tuple[float, float]): Factor range for contrast adjustment.
            zoom_range (Tuple[float, float]): Scale range for random zoom.
            p (float): Probability of applying each augmentation step.
        """
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.zoom_range = zoom_range
        self.p = p

    def apply_horizontal_flip(self, img: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Flip image and mask horizontally."""
        img_flipped = cv2.flip(img, 1)
        mask_flipped = cv2.flip(mask, 1) if mask is not None else None
        return img_flipped, mask_flipped

    def apply_vertical_flip(self, img: np.ndarray, mask: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Flip image and mask vertically."""
        img_flipped = cv2.flip(img, 0)
        mask_flipped = cv2.flip(mask, 0) if mask is not None else None
        return img_flipped, mask_flipped

    def apply_rotation(
        self, img: np.ndarray, mask: Optional[np.ndarray], angle: float
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Rotate image and mask by exact angle around center."""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        img_rot = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask_rot = None
        if mask is not None:
            mask_rot = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img_rot, mask_rot

    def apply_zoom(
        self, img: np.ndarray, mask: Optional[np.ndarray], scale: float
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply random zoom centered on image."""
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, 0, scale)

        img_zoomed = cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask_zoomed = None
        if mask is not None:
            mask_zoomed = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return img_zoomed, mask_zoomed

    def apply_brightness_contrast(
        self, img: np.ndarray, brightness_factor: float, contrast_factor: float
    ) -> np.ndarray:
        """
        Adjust image intensity (applied ONLY to CT slice image, NOT binary defect mask).
        """
        img_adj = img.astype(np.float32)
        # Apply contrast
        mean = np.mean(img_adj)
        img_adj = (img_adj - mean) * contrast_factor + mean
        # Apply brightness
        img_adj = img_adj * brightness_factor
        return np.clip(img_adj, 0.0, 1.0 if img.max() <= 1.0 else 255.0).astype(img.dtype)

    def augment(
        self, image: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply random augmentation pipeline to image and optional mask.
        
        Args:
            image (np.ndarray): Input CT image array (H, W) or (H, W, C).
            mask (np.ndarray, optional): Ground truth binary mask array (H, W) or (H, W, 1).
            
        Returns:
            Tuple[np.ndarray, Optional[np.ndarray]]: Augmented (image, mask).
        """
        # TODO: Teammate Dataset Integration - Connect to tf.data.Dataset map function during batch generator loading.

        augmented_img = image.copy()
        augmented_mask = mask.copy() if mask is not None else None

        # 1. Horizontal Flip
        if self.horizontal_flip and random.random() < self.p:
            augmented_img, augmented_mask = self.apply_horizontal_flip(augmented_img, augmented_mask)

        # 2. Vertical Flip
        if self.vertical_flip and random.random() < self.p:
            augmented_img, augmented_mask = self.apply_vertical_flip(augmented_img, augmented_mask)

        # 3. Rotation
        if self.rotation_range > 0 and random.random() < self.p:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            augmented_img, augmented_mask = self.apply_rotation(augmented_img, augmented_mask, angle)

        # 4. Random Zoom
        if self.zoom_range != (1.0, 1.0) and random.random() < self.p:
            scale = random.uniform(self.zoom_range[0], self.zoom_range[1])
            augmented_img, augmented_mask = self.apply_zoom(augmented_img, augmented_mask, scale)

        # 5. Brightness & Contrast (CT Image only)
        if random.random() < self.p:
            b_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
            c_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
            augmented_img = self.apply_brightness_contrast(augmented_img, b_factor, c_factor)

        # Ensure channel dimensions are preserved
        if len(image.shape) == 3 and len(augmented_img.shape) == 2:
            augmented_img = np.expand_dims(augmented_img, axis=-1)
        if mask is not None and len(mask.shape) == 3 and len(augmented_mask.shape) == 2:
            augmented_mask = np.expand_dims(augmented_mask, axis=-1)

        return augmented_img, augmented_mask


if __name__ == "__main__":
    print("InSight3D Data Augmentation Module Test")
    
    dummy_img = np.random.rand(256, 256, 1).astype(np.float32)
    dummy_mask = np.zeros((256, 256, 1), dtype=np.float32)
    # Add a rectangle defect on mask
    dummy_mask[80:120, 80:120, 0] = 1.0

    augmentor = DefectDataAugmentor(rotation_range=25.0, zoom_range=(0.8, 1.2), p=1.0)
    aug_img, aug_mask = augmentor.augment(dummy_img, dummy_mask)

    print(f"Augmented Image shape: {aug_img.shape}, min={aug_img.min():.2f}, max={aug_img.max():.2f}")
    print(f"Augmented Mask shape: {aug_mask.shape}, sum of defect pixels: {np.sum(aug_mask)}")
