"""
InSight3D - Preprocessing Module
AI-Based Internal Defect Segmentation in 3D-Printed Electronic Components Using U-Net

Functions for CT slice image preprocessing:
- Resizing (e.g. 256x256)
- Pixel Normalization (0-255 -> 0-1)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Noise Removal (Gaussian Blur, Median Blur)
- Output File Persistence
"""

import os
from typing import Tuple, Optional, Union
import cv2
import numpy as np


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    is_mask: bool = False
) -> np.ndarray:
    """
    Resize image to target dimensions.
    
    Args:
        image (np.ndarray): Input image or mask array.
        target_size (Tuple[int, int]): Desired (height, width).
        is_mask (bool): If True, uses nearest neighbor interpolation to preserve categorical label values.
        
    Returns:
        np.ndarray: Resized image.
    """
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    # cv2.resize expects (width, height) tuple
    dsize = (target_size[1], target_size[0])
    resized = cv2.resize(image, dsize, interpolation=interpolation)
    return resized


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values from [0, 255] range to [0.0, 1.0].
    
    Args:
        image (np.ndarray): Input uint8 or float image.
        
    Returns:
        np.ndarray: Normalized float32 image in range [0.0, 1.0].
    """
    img_float = image.astype(np.float32)
    if img_float.max() > 1.0:
        img_float /= 255.0
    return np.clip(img_float, 0.0, 1.0)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance internal CT slice contrast.
    
    Args:
        image (np.ndarray): Grayscale image (uint8 or uint16).
        clip_limit (float): Threshold for contrast limiting.
        tile_grid_size (Tuple[int, int]): Size of grid for histogram equalization.
        
    Returns:
        np.ndarray: Contrast-enhanced grayscale image (uint8).
    """
    # Ensure image is in uint8 format for OpenCV CLAHE
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            img_uint8 = (image * 255.0).astype(np.uint8)
        else:
            img_uint8 = image.astype(np.uint8)
    else:
        img_uint8 = image.copy()

    # Convert to single channel grayscale if multi-channel
    if len(img_uint8.shape) == 3 and img_uint8.shape[2] == 3:
        img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(img_uint8)
    return enhanced


def remove_noise(
    image: np.ndarray,
    method: Optional[str] = "gaussian",
    kernel_size: int = 3
) -> np.ndarray:
    """
    Apply noise removal techniques (Gaussian Blur or Median Blur).
    
    Args:
        image (np.ndarray): Input image array.
        method (str, optional): 'gaussian', 'median', or None.
        kernel_size (int): Size of kernel (must be positive odd integer, e.g. 3, 5).
        
    Returns:
        np.ndarray: Noise-filtered image.
    """
    if method is None or method.lower() == "none":
        return image.copy()

    # Enforce odd kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1

    method = method.lower()
    if method == "gaussian":
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    elif method == "median":
        return cv2.medianBlur(image, kernel_size)
    else:
        raise ValueError(f"Unsupported noise removal method: {method}. Choose 'gaussian', 'median', or None.")


def preprocess_pipeline(
    image: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    is_mask: bool = False,
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    noise_reduction: Optional[str] = "gaussian",
    kernel_size: int = 3,
    normalize: bool = True
) -> np.ndarray:
    """
    Complete Preprocessing Pipeline for X-ray CT slice image or mask.
    
    Processing Steps:
    1. Resizing to target_size (e.g. 256x256).
    2. (If CT Image & Enabled) Contrast enhancement via CLAHE.
    3. (If CT Image & Enabled) Noise reduction via Gaussian/Median blur.
    4. Pixel normalization to [0.0, 1.0].
    
    Args:
        image (np.ndarray): Raw CT image or segmentation mask.
        target_size (Tuple[int, int]): Desired dimensions (H, W).
        is_mask (bool): True if processing ground truth defect mask.
        use_clahe (bool): Whether to apply CLAHE enhancement.
        clahe_clip (float): CLAHE clip limit.
        noise_reduction (str, optional): 'gaussian', 'median', or None.
        kernel_size (int): Kernel size for noise reduction.
        normalize (bool): Whether to scale values to [0.0, 1.0].
        
    Returns:
        np.ndarray: Preprocessed image tensor array.
    """
    # TODO: Teammate Dataset Integration - Adjust channel handling if multi-modal or 16-bit CT DICOM inputs are collected.
    
    # 1. Resize
    processed = resize_image(image, target_size=target_size, is_mask=is_mask)

    if is_mask:
        # For segmentation masks, ensure binary values (0 or 1)
        if processed.max() > 1:
            processed = (processed > 127).astype(np.float32)
        else:
            processed = processed.astype(np.float32)
        if len(processed.shape) == 2:
            processed = np.expand_dims(processed, axis=-1)
        return processed

    # 2. CLAHE (For grayscale CT slice image)
    if use_clahe:
        processed = apply_clahe(processed, clip_limit=clahe_clip)

    # 3. Noise Removal
    if noise_reduction:
        processed = remove_noise(processed, method=noise_reduction, kernel_size=kernel_size)

    # 4. Normalize
    if normalize:
        processed = normalize_image(processed)

    # Ensure single channel or 3-channel layout
    if len(processed.shape) == 2:
        processed = np.expand_dims(processed, axis=-1)

    return processed


def save_processed_pair(
    processed_img: np.ndarray,
    processed_mask: Optional[np.ndarray],
    output_img_dir: str = "results/processed_images",
    output_mask_dir: str = "results/processed_masks",
    filename: str = "sample_001.png"
) -> Tuple[str, Optional[str]]:
    """
    Save preprocessed image and optional mask to designated directories.
    
    Args:
        processed_img (np.ndarray): Processed CT image (float [0,1] or uint8).
        processed_mask (np.ndarray, optional): Processed binary mask.
        output_img_dir (str): Directory path to save image.
        output_mask_dir (str): Directory path to save mask.
        filename (str): Base output filename.
        
    Returns:
        Tuple[str, Optional[str]]: Paths where image and mask were saved.
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)

    img_save_path = os.path.join(output_img_dir, filename)
    
    # Convert float [0, 1] to uint8 [0, 255] for image saving
    if processed_img.dtype in [np.float32, np.float64] and processed_img.max() <= 1.0:
        save_img = (processed_img * 255.0).astype(np.uint8)
    else:
        save_img = processed_img.astype(np.uint8)
        
    cv2.imwrite(img_save_path, save_img)

    mask_save_path = None
    if processed_mask is not None:
        mask_save_path = os.path.join(output_mask_dir, filename)
        if processed_mask.dtype in [np.float32, np.float64] and processed_mask.max() <= 1.0:
            save_mask = (processed_mask * 255.0).astype(np.uint8)
        else:
            save_mask = processed_mask.astype(np.uint8)
        cv2.imwrite(mask_save_path, save_mask)

    return img_save_path, mask_save_path


if __name__ == "__main__":
    print("InSight3D Preprocessing Module Test")
    
    # Create a synthetic CT slice test image (200x200 uint8)
    dummy_ct = np.random.randint(50, 200, size=(200, 200), dtype=np.uint8)
    # Add a synthetic void defect region
    cv2.circle(dummy_ct, (100, 100), 20, 10, -1)

    # Process using pipeline
    processed = preprocess_pipeline(dummy_ct, target_size=(256, 256), use_clahe=True, noise_reduction="gaussian")
    print(f"Original shape: {dummy_ct.shape} -> Processed shape: {processed.shape}")
    print(f"Value range: min={processed.min():.4f}, max={processed.max():.4f}, dtype={processed.dtype}")
    
    img_p, _ = save_processed_pair(processed, None, filename="test_ct_slice.png")
    print(f"Saved processed output to: {img_p}")
