"""
InSight3D - Dataset Preprocessing & Train/Val/Test Splitting Module
AI-Based Internal Defect Segmentation in 3D-Printed Electronic Components Using U-Net

Workflow:
1. Scan raw CT slice images from `dataset/XmCT_sample_S01`.
2. Extract component body boundaries & internal defect masks (voids, porosity, delamination).
3. Preprocess and resize images and masks to 256x256.
4. Shuffle and split into Train (70%), Validation (15%), and Test (15%).
5. Save paired images and masks to `dataset/train`, `dataset/val`, and `dataset/test`.
"""

import os
import sys
import glob
import random
from typing import Tuple, List
import cv2
import numpy as np

# Ensure parent directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocess import preprocess_pipeline


def extract_ct_defect_mask(raw_img: np.ndarray) -> np.ndarray:
    """
    Extract internal defect mask from X-ray CT slice image.
    
    1. Threshold & Morphological closing to find component body mask.
    2. Identify internal dark regions (voids, pores, unmaterialized channels) within component body.
    
    Args:
        raw_img (np.ndarray): Grayscale CT slice image array (924x924 or any size).
        
    Returns:
        np.ndarray: Binary mask array (255 for defect pixels, 0 for background/solid).
    """
    # 1. Component Body Segmentation (Otsu threshold + Morphological closing)
    _, thresh = cv2.threshold(raw_img, 60, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    body_mask = np.zeros_like(raw_img)
    if contours:
        c = max(contours, key=cv2.contourArea)
        cv2.drawContours(body_mask, [c], -1, 255, -1)

    # 2. Defect Void Segmentation inside Component Body
    # In CT, solid printed material is bright (~140-220), internal voids/pores are dark (<100)
    internal_voids = (raw_img < 105) & (body_mask == 255)
    defect_mask = np.zeros_like(raw_img)
    defect_mask[internal_voids] = 255

    # Filter out single-pixel noise with opening
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    defect_mask = cv2.morphologyEx(defect_mask, cv2.MORPH_OPEN, small_kernel)

    return defect_mask


def prepare_and_split_dataset(
    source_dir: str = "dataset/XmCT_sample_S01",
    target_dataset_dir: str = "dataset",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    target_size: Tuple[int, int] = (256, 256),
    seed: int = 42
) -> dict:
    """
    Process raw CT slice images, generate defect masks, and save train/val/test splits.
    """
    random.seed(seed)
    np.random.seed(seed)

    # Find BMP image files
    img_paths = sorted(glob.glob(os.path.join(source_dir, "*.bmp")))
    if not img_paths:
        img_paths = sorted(glob.glob(os.path.join(source_dir, "*.png"))) + sorted(glob.glob(os.path.join(source_dir, "*.tif")))

    if not img_paths:
        raise FileNotFoundError(f"No CT image files found in directory: {source_dir}")

    print(f"[INFO] Total raw CT slice images found in '{source_dir}': {len(img_paths)}")

    # Shuffle paths reproducibly
    shuffled_paths = img_paths.copy()
    random.shuffle(shuffled_paths)

    total_count = len(shuffled_paths)
    n_train = int(total_count * train_ratio)
    n_val = int(total_count * val_ratio)
    n_test = total_count - n_train - n_val

    train_paths = shuffled_paths[:n_train]
    val_paths = shuffled_paths[n_train:n_train + n_val]
    test_paths = shuffled_paths[n_train + n_val:]

    print(f"[INFO] Train Split: {len(train_paths)} samples ({train_ratio*100:.0f}%)")
    print(f"[INFO] Val Split:   {len(val_paths)} samples ({val_ratio*100:.0f}%)")
    print(f"[INFO] Test Split:  {len(test_paths)} samples ({test_ratio*100:.0f}%)")

    splits = {
        "train": train_paths,
        "val": val_paths,
        "test": test_paths
    }

    stats = {}

    for split_name, paths in splits.items():
        img_out_dir = os.path.join(target_dataset_dir, split_name, "images")
        mask_out_dir = os.path.join(target_dataset_dir, split_name, "masks")

        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(mask_out_dir, exist_ok=True)

        saved_count = 0
        for p in paths:
            raw = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if raw is None:
                continue

            # Extract defect mask
            mask_raw = extract_ct_defect_mask(raw)

            # Preprocess & Resize to 256x256
            prep_img = preprocess_pipeline(raw, target_size=target_size, is_mask=False, use_clahe=True, noise_reduction="gaussian")
            prep_mask = preprocess_pipeline(mask_raw, target_size=target_size, is_mask=True)

            base_name = os.path.splitext(os.path.basename(p))[0] + ".png"
            img_save_path = os.path.join(img_out_dir, base_name)
            mask_save_path = os.path.join(mask_out_dir, base_name)

            # Save as PNG uint8
            cv2.imwrite(img_save_path, (prep_img[:, :, 0] * 255.0).astype(np.uint8))
            cv2.imwrite(mask_save_path, (prep_mask[:, :, 0] * 255.0).astype(np.uint8))

            saved_count += 1

        stats[split_name] = saved_count
        print(f"[SUCCESS] Saved {saved_count} paired (image, mask) samples into 'dataset/{split_name}'")

    return stats


if __name__ == "__main__":
    print("InSight3D - Preparing and Splitting Dataset...")
    results = prepare_and_split_dataset()
    print("Dataset Preparation Completed Successfully!", results)
