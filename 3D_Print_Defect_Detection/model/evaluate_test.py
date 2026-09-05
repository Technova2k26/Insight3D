"""
InSight3D - Test Dataset Evaluation Module
AI-Based Internal Defect Segmentation in 3D-Printed Electronic Components Using U-Net

Workflow:
1. Load best trained U-Net checkpoint from `models/best_model.pth`.
2. Evaluate performance on unseen `dataset/test` split (63 CT slice images).
3. Compute test performance metrics:
   - Dice Coefficient
   - IoU (Intersection over Union)
   - Precision & Recall
   - Binary Accuracy
   - BCE-Dice Loss
4. Export test results to JSON and CSV reports (`results/reports/test_evaluation_report.csv`).
5. Render and save a visual 4x4 test predictions grid (`results/reports/test_predictions_grid.png`).
"""

import os
import sys
import glob
import json
from typing import Dict, Any
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.unet_torch import UNetTorch
from model.tune_and_train import FastCTDefectDataset, DiceBCELoss, evaluate
from utils.metrics import dice_coef, iou_score, precision_m, recall_m


def evaluate_test_set(
    model_path: str = "models/best_model.pth",
    dataset_dir: str = "dataset",
    reports_dir: str = "results/reports",
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Run full test evaluation on dataset/test split and generate metrics report.
    """
    os.makedirs(reports_dir, exist_ok=True)
    test_dir = os.path.join(dataset_dir, "test")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model weights checkpoint not found at: {model_path}")

    # Load Model Checkpoint
    print(f"[INFO] Loading best U-Net checkpoint from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    init_filters = checkpoint.get("init_filters", 32)
    dropout_rate = checkpoint.get("dropout_rate", 0.1)

    model = UNetTorch(in_channels=1, out_channels=1, init_filters=init_filters, dropout_rate=dropout_rate).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_ds = FastCTDefectDataset(test_dir)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    criterion = DiceBCELoss()

    # 1. Compute Quantitative Metrics
    print(f"[INFO] Evaluating model on {len(test_ds)} test CT slice images...")
    metrics = evaluate(model, test_loader, criterion, device)

    print("\n" + "=" * 60)
    print("TEST DATASET PERFORMANCE EVALUATION RESULTS:")
    print("=" * 60)
    print(f"  - Test Dice Coefficient: {metrics['val_dice']:.4f}")
    print(f"  - Test IoU Score:        {metrics['val_iou']:.4f}")
    print(f"  - Test Precision:        {metrics['val_precision']:.4f}")
    print(f"  - Test Recall:           {metrics['val_recall']:.4f}")
    print(f"  - Test Loss:             {metrics['val_loss']:.4f}")
    print("=" * 60 + "\n")

    test_report = {
        "dataset_split": "test",
        "sample_count": len(test_ds),
        "test_dice_coef": metrics["val_dice"],
        "test_iou_score": metrics["val_iou"],
        "test_precision": metrics["val_precision"],
        "test_recall": metrics["val_recall"],
        "test_bce_dice_loss": metrics["val_loss"]
    }

    # Save Reports
    json_path = os.path.join(reports_dir, "test_evaluation_report.json")
    csv_path = os.path.join(reports_dir, "test_evaluation_report.csv")

    with open(json_path, "w") as f:
        json.dump(test_report, f, indent=4)

    pd.DataFrame([test_report]).to_csv(csv_path, index=False)
    print(f"[SUCCESS] Test evaluation report saved to: {csv_path}")

    # 2. Render Test Predictions Grid Overlay
    grid_path = os.path.join(reports_dir, "test_predictions_grid.png")
    render_test_grid(model, test_ds, grid_path, device=device)

    return test_report


def render_test_grid(model, test_ds, save_path: str, num_samples: int = 4, device: str = "cpu"):
    """
    Render 4-sample comparison grid showing Original CT, Ground Truth Mask, and U-Net Prediction Overlay.
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 3 * num_samples))
    fig.suptitle("InSight3D - Test Set Prediction Overlays", fontsize=14, fontweight='bold', y=0.99)

    model.eval()
    indices = np.linspace(0, len(test_ds) - 1, num_samples, dtype=int)

    for idx, sample_i in enumerate(indices):
        tensor_img, tensor_mask = test_ds[sample_i]
        with torch.no_grad():
            inp = tensor_img.unsqueeze(0).to(device)
            prob_map = model(inp).squeeze().cpu().numpy()

        img_np = tensor_img.squeeze().numpy()
        gt_mask_np = tensor_mask.squeeze().numpy()

        binary_pred = (prob_map >= 0.5).astype(np.uint8)

        base = (img_np * 255.0).astype(np.uint8)
        base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
        overlay = base_bgr.copy()
        overlay[binary_pred == 1] = [0, 0, 255]
        overlay = cv2.addWeighted(base_bgr, 0.5, overlay, 0.5, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        axes[idx, 0].imshow(img_np, cmap='gray')
        axes[idx, 0].set_title(f"Test Sample #{sample_i+1} - Original CT", fontsize=10)
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(gt_mask_np, cmap='inferno')
        axes[idx, 1].set_title("Ground Truth Defect Mask", fontsize=10)
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(overlay_rgb)
        axes[idx, 2].set_title(f"Predicted Overlay (Dice: {dice_coef(gt_mask_np, binary_pred):.2f})", fontsize=10)
        axes[idx, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[SUCCESS] Saved test predictions grid image to: {save_path}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_test_set(device=device)
