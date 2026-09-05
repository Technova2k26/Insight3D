"""
InSight3D - U-Net Training Pipeline
AI-Based Internal Defect Segmentation in 3D-Printed Electronic Components Using U-Net

Workflow:
1. Load dataset paths from `dataset/train` and `dataset/val`.
2. Handle empty dataset directories gracefully with TODO warnings & placeholder loaders.
3. Build & Compile U-Net Architecture.
4. Setup Callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard.
5. Execute Model Training & Save `best_model.keras`.
"""

import os
import sys
import glob
import argparse
from typing import Tuple, List, Optional
import numpy as np
import tensorflow as tf

# Ensure parent directory is in sys.path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.preprocess import preprocess_pipeline
from model.unet import build_unet
from utils.metrics import bce_dice_loss, dice_coef, iou_score, precision_m, recall_m


def get_image_mask_paths(split_dir: str) -> Tuple[List[str], List[str]]:
    """
    Search for image and mask file paths in split directory.
    
    Args:
        split_dir (str): Path to split folder e.g., 'dataset/train'.
        
    Returns:
        Tuple[List[str], List[str]]: Lists of image paths and mask paths.
    """
    # TODO: Teammate Dataset Integration - Update file extensions if DICOM (.dcm), TIFF (.tif), or NIfTI (.nii) files are collected.
    valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    
    img_dir = os.path.join(split_dir, "images")
    mask_dir = os.path.join(split_dir, "masks")
    
    img_paths = []
    mask_paths = []
    
    if os.path.exists(img_dir):
        for ext in valid_exts:
            img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
            img_paths.extend(glob.glob(os.path.join(img_dir, ext.upper())))
            
    if os.path.exists(mask_dir):
        for ext in valid_exts:
            mask_paths.extend(glob.glob(os.path.join(mask_dir, ext)))
            mask_paths.extend(glob.glob(os.path.join(mask_dir, ext.upper())))

    img_paths.sort()
    mask_paths.sort()
    return img_paths, mask_paths


class DatasetSequence(tf.keras.utils.Sequence):
    """
    Custom Keras Data Generator for batching CT slice images and defect masks.
    """

    def __init__(
        self,
        img_paths: List[str],
        mask_paths: List[str],
        batch_size: int = 8,
        img_size: Tuple[int, int] = (256, 256),
        is_training: bool = True
    ):
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.batch_size = batch_size
        self.img_size = img_size
        self.is_training = is_training
        
        # TODO: Teammate Dataset Integration - Instantiate DefectDataAugmentor here if live online augmentation during training is desired.

    def __len__(self):
        return int(np.ceil(len(self.img_paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_img_paths = self.img_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_mask_paths = self.mask_paths[idx * self.batch_size : (idx + 1) * self.batch_size]

        images = []
        masks = []

        for img_p, mask_p in zip(batch_img_paths, batch_mask_paths):
            import cv2
            raw_img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
            raw_mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

            if raw_img is None:
                raise ValueError(f"Failed to read CT image at: {img_p}")
            if raw_mask is None:
                raise ValueError(f"Failed to read defect mask at: {mask_p}")

            p_img = preprocess_pipeline(raw_img, target_size=self.img_size, is_mask=False)
            p_mask = preprocess_pipeline(raw_mask, target_size=self.img_size, is_mask=True)

            images.append(p_img)
            masks.append(p_mask)

        return np.array(images, dtype=np.float32), np.array(masks, dtype=np.float32)


def generate_synthetic_demo_data(num_samples: int = 20, img_size: Tuple[int, int] = (256, 256)):
    """
    Generate synthetic CT images and masks for offline pipeline testing before teammate dataset collection finishes.
    """
    import cv2
    X = []
    Y = []
    for _ in range(num_samples):
        # Create background noise simulating electronic component CT slice
        img = np.random.normal(128, 20, (img_size[0], img_size[1])).astype(np.float32)
        mask = np.zeros((img_size[0], img_size[1]), dtype=np.float32)
        
        # Add 1 to 3 random defect voids (dark spots)
        num_defects = np.random.randint(1, 4)
        for _ in range(num_defects):
            cx, cy = np.random.randint(40, 216, size=2)
            radius = np.random.randint(5, 20)
            cv2.circle(img, (cx, cy), radius, float(np.random.randint(10, 40)), -1)
            cv2.circle(mask, (cx, cy), radius, 1.0, -1)
            
        p_img = preprocess_pipeline(img.astype(np.uint8), target_size=img_size, is_mask=False)
        p_mask = preprocess_pipeline(mask.astype(np.uint8), target_size=img_size, is_mask=True)
        
        X.append(p_img)
        Y.append(p_mask)
        
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


def train_model(
    dataset_dir: str = "dataset",
    output_model_dir: str = "models",
    log_dir: str = "logs",
    epochs: int = 25,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    img_size: Tuple[int, int] = (256, 256),
    init_filters: int = 32,
    use_synthetic_if_empty: bool = True
):
    """
    Train U-Net Defect Segmentation Model.
    """
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    train_img_paths, train_mask_paths = get_image_mask_paths(os.path.join(dataset_dir, "train"))
    val_img_paths, val_mask_paths = get_image_mask_paths(os.path.join(dataset_dir, "val"))

    print(f"[INFO] Training CT Slice Images Found: {len(train_img_paths)}")
    print(f"[INFO] Validation CT Slice Images Found: {len(val_img_paths)}")

    using_synthetic = False
    if len(train_img_paths) == 0 or len(train_mask_paths) == 0:
        print("\n" + "=" * 80)
        print("[WARNING] Dataset directory 'dataset/train' is currently EMPTY or incomplete!")
        print("          # TODO: Teammate is collecting the 3D printed CT slice dataset.")
        print("          Place image files in: dataset/train/images/")
        print("          Place mask files in:  dataset/train/masks/")
        print("=" * 80 + "\n")
        
        if use_synthetic_if_empty:
            print("[INFO] Generating synthetic CT slice data to verify training pipeline & save initial model baseline...")
            X_train, Y_train = generate_synthetic_demo_data(num_samples=32, img_size=img_size)
            X_val, Y_val = generate_synthetic_demo_data(num_samples=8, img_size=img_size)
            using_synthetic = True
        else:
            print("[ABORT] Cannot proceed without training samples. Please populate dataset or set --use-synthetic.")
            return None

    # Build U-Net
    print(f"\n[INFO] Building U-Net Architecture (Input shape: ({img_size[0]}, {img_size[1]}, 1), Base filters: {init_filters})...")
    model = build_unet(input_shape=(img_size[0], img_size[1], 1), init_filters=init_filters)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=bce_dice_loss,
        metrics=[dice_coef, iou_score, precision_m, recall_m, "accuracy"]
    )

    # Define Training Callbacks
    best_model_path = os.path.join(output_model_dir, "best_model.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_dice_coef",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1
        )
    ]

    print(f"[INFO] Starting Model Training for {epochs} epochs...")
    if using_synthetic:
        history = model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
    else:
        train_gen = DatasetSequence(train_img_paths, train_mask_paths, batch_size=batch_size, img_size=img_size)
        val_gen = DatasetSequence(val_img_paths, val_mask_paths, batch_size=batch_size, img_size=img_size)
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

    print(f"\n[SUCCESS] Training completed successfully!")
    print(f"[SUCCESS] Best U-Net weights saved to: {best_model_path}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InSight3D - Train U-Net Defect Segmentation Model")
    parser.add_argument("--dataset-dir", type=str, default="dataset", help="Root path to dataset folder")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory to save trained model")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=256, help="Target image height and width")
    
    args = parser.parse_args()
    
    train_model(
        dataset_dir=args.dataset_dir,
        output_model_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        img_size=(args.img_size, args.img_size)
    )
