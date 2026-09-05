"""
InSight3D - Hyperparameter Tuning & Resume/Continue U-Net Model Training Pipeline
AI-Based Internal Defect Segmentation in 3D-Printed Electronic Components Using U-Net

Workflow:
1. Automatically resolve dataset paths (`dataset/train` and `dataset/val`).
2. Support `--resume`: Load existing weights from `models/best_model.pth` and continue training.
3. Support `--epochs <N>`: Specify total/additional epochs to train.
4. Exception-safe real-time terminal output showing Epoch, Train Loss, Val Loss, Val Dice, Val Accuracy, Val Recall & Precision.
5. Save updated best model weights to `models/best_model.pth` and update history plots.
"""

import os
import sys
import glob
import json
import time
import argparse
import traceback
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Absolute Project Root Directory Resolution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from model.unet_torch import UNetTorch
from utils.metrics import dice_coef, iou_score, precision_m, recall_m


class FastCTDefectDataset(Dataset):
    """Fast PyTorch Dataset reading preprocessed 256x256 image & mask arrays directly."""

    def __init__(self, split_dir: str):
        self.img_dir = os.path.join(split_dir, "images")
        self.mask_dir = os.path.join(split_dir, "masks")

        self.img_paths = sorted(glob.glob(os.path.join(self.img_dir, "*.png")) + glob.glob(os.path.join(self.img_dir, "*.jpg")))
        self.mask_paths = sorted(glob.glob(os.path.join(self.mask_dir, "*.png")) + glob.glob(os.path.join(self.mask_dir, "*.jpg")))

        if len(self.img_paths) == 0:
            raise ValueError(f"No image files found in dataset path: '{self.img_dir}'")
        if len(self.img_paths) != len(self.mask_paths):
            raise ValueError(f"Mismatch between images ({len(self.img_paths)}) and masks ({len(self.mask_paths)}) in {split_dir}")

        self.images = []
        self.masks = []

        for img_p, mask_p in zip(self.img_paths, self.mask_paths):
            img = cv2.imread(img_p, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)

            img_f = (img.astype(np.float32) / 255.0) if img.max() > 1 else img.astype(np.float32)
            mask_f = (mask.astype(np.float32) / 255.0) if mask.max() > 1 else mask.astype(np.float32)

            self.images.append(torch.tensor(img_f, dtype=torch.float32).unsqueeze(0))
            self.masks.append(torch.tensor(mask_f, dtype=torch.float32).unsqueeze(0))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.masks[idx]


class DiceBCELoss(nn.Module):
    """Combined Binary Cross-Entropy and Dice Loss for PyTorch U-Net."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCELoss()

    def forward(self, y_pred, y_true):
        y_pred_clamped = torch.clamp(y_pred, 1e-7, 1.0 - 1e-7)
        y_true_clamped = torch.clamp(y_true, 0.0, 1.0)
        
        bce_loss = self.bce(y_pred_clamped, y_true_clamped)
        
        y_pred_f = y_pred_clamped.view(-1)
        y_true_f = y_true_clamped.view(-1)
        intersection = (y_pred_f * y_true_f).sum()
        dice_score = (2.0 * intersection + self.smooth) / (y_pred_f.sum() + y_true_f.sum() + self.smooth)
        dice_loss = 1.0 - dice_score

        return bce_loss + dice_loss


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in dataloader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_masks = []
    with torch.no_grad():
        for imgs, masks in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            total_loss += loss.item() * imgs.size(0)

            all_preds.append(preds.cpu().numpy())
            all_masks.append(masks.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    binary_preds = (all_preds >= 0.5).astype(np.float32)
    binary_masks = (all_masks >= 0.5).astype(np.float32)

    try:
        d_score = float(dice_coef(binary_masks, binary_preds))
        i_score = float(iou_score(binary_masks, binary_preds))
        p_score = float(precision_m(binary_masks, binary_preds))
        r_score = float(recall_m(binary_masks, binary_preds))
        acc_score = float(np.mean(binary_masks == binary_preds)) * 100.0
    except Exception:
        d_score, i_score, p_score, r_score, acc_score = 0.0, 0.0, 0.0, 0.0, 0.0

    val_loss = total_loss / len(dataloader.dataset)

    return {
        "val_loss": round(val_loss, 4),
        "val_dice": round(d_score, 4),
        "val_iou": round(i_score, 4),
        "val_precision": round(p_score * 100.0, 2),
        "val_recall": round(r_score * 100.0, 2),
        "val_accuracy": round(acc_score, 2)
    }


def run_hyperparameter_tuning(
    dataset_dir: str = os.path.join(PROJECT_ROOT, "dataset"),
    reports_dir: str = os.path.join(PROJECT_ROOT, "results", "reports"),
    device: str = "cpu"
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dataset, Dataset]:
    """
    Execute Grid Search Hyperparameter Tuning across learning rates, filter counts, and batch sizes.
    """
    os.makedirs(reports_dir, exist_ok=True)

    print(f"[INFO] Loading train and validation datasets from: '{dataset_dir}'", flush=True)
    train_ds = FastCTDefectDataset(os.path.join(dataset_dir, "train"))
    val_ds = FastCTDefectDataset(os.path.join(dataset_dir, "val"))

    hyperparam_grid = [
        {"lr": 1e-4, "init_filters": 16, "batch_size": 16, "dropout_rate": 0.1},
        {"lr": 5e-4, "init_filters": 16, "batch_size": 16, "dropout_rate": 0.1},
        {"lr": 1e-3, "init_filters": 32, "batch_size": 16, "dropout_rate": 0.1},
        {"lr": 5e-4, "init_filters": 32, "batch_size": 16, "dropout_rate": 0.2},
        {"lr": 1e-4, "init_filters": 32, "batch_size": 16, "dropout_rate": 0.1}
    ]

    print("\n" + "=" * 80, flush=True)
    print("STARTING HYPERPARAMETER TUNING SEARCH (5 TRIAL CONFIGURATIONS)", flush=True)
    print("=" * 80, flush=True)

    tuning_results = []
    best_dice = -1.0
    best_config = None

    criterion = DiceBCELoss()

    for idx, config in enumerate(hyperparam_grid, start=1):
        print(f"\n--- Trial {idx}/{len(hyperparam_grid)}: LR={config['lr']}, Filters={config['init_filters']}, BatchSize={config['batch_size']}, Dropout={config['dropout_rate']} ---", flush=True)
        
        train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

        model = UNetTorch(in_channels=1, out_channels=1, init_filters=config['init_filters'], dropout_rate=config['dropout_rate']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

        for epoch in range(1, 4):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        val_metrics = evaluate(model, val_loader, criterion, device)
        
        trial_summary = {
            "trial_id": idx,
            "learning_rate": config['lr'],
            "init_filters": config['init_filters'],
            "batch_size": config['batch_size'],
            "dropout_rate": config['dropout_rate'],
            "val_loss": val_metrics["val_loss"],
            "val_dice": val_metrics["val_dice"],
            "val_iou": val_metrics["val_iou"],
            "val_precision": val_metrics["val_precision"],
            "val_recall": val_metrics["val_recall"],
            "val_accuracy": val_metrics["val_accuracy"]
        }
        
        tuning_results.append(trial_summary)
        print(f"Trial {idx} Results -> Val Loss: {val_metrics['val_loss']}, Val Dice: {val_metrics['val_dice']}, Val IoU: {val_metrics['val_iou']}, Accuracy: {val_metrics['val_accuracy']}%, Recall: {val_metrics['val_recall']}%", flush=True)

        if val_metrics["val_dice"] > best_dice:
            best_dice = val_metrics["val_dice"]
            best_config = config.copy()

    tuning_json_path = os.path.join(reports_dir, "hyperparameter_tuning_results.json")
    tuning_csv_path = os.path.join(reports_dir, "hyperparameter_tuning_results.csv")

    with open(tuning_json_path, "w") as f:
        json.dump(tuning_results, f, indent=4)

    df_tuning = pd.DataFrame(tuning_results)
    df_tuning.to_csv(tuning_csv_path, index=False)

    print("\n" + "=" * 80, flush=True)
    print(f"HYPERPARAMETER TUNING COMPLETED! BEST CONFIGURATION SELECTED:", flush=True)
    print(json.dumps(best_config, indent=4), flush=True)
    print(f"Saved hyperparameter tuning results to: {tuning_csv_path}", flush=True)
    print("=" * 80 + "\n", flush=True)

    return best_config, tuning_results, train_ds, val_ds


def train_best_model(
    best_config: Dict[str, Any],
    train_ds: Dataset,
    val_ds: Dataset,
    output_model_dir: str = os.path.join(PROJECT_ROOT, "models"),
    reports_dir: str = os.path.join(PROJECT_ROOT, "results", "reports"),
    epochs: int = 15,
    resume: bool = False,
    device: str = "cpu"
) -> Tuple[str, Dict[str, List[float]]]:
    """
    Train full U-Net model using selected BEST hyperparameter configuration.
    """
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    train_loader = DataLoader(train_ds, batch_size=best_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=best_config['batch_size'], shuffle=False)

    model = UNetTorch(
        in_channels=1,
        out_channels=1,
        init_filters=best_config['init_filters'],
        dropout_rate=best_config['dropout_rate']
    ).to(device)

    best_model_path = os.path.join(output_model_dir, "best_model.pth")
    best_val_dice = -1.0

    if resume and os.path.exists(best_model_path):
        print(f"[INFO] Resuming training from checkpoint: '{best_model_path}'", flush=True)
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_val_dice = checkpoint.get("best_val_dice", -1.0)
        print(f"[INFO] Existing best Val Dice score: {best_val_dice:.4f}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=best_config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4)
    criterion = DiceBCELoss()

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_accuracy": [],
        "val_recall": [],
        "val_precision": []
    }

    print(f"\n[INFO] Starting Model Training ({epochs} Additional Epochs)...", flush=True)
    print("=" * 110, flush=True)
    print(f"{'Epoch':<10} | {'Elapsed':<8} | {'Train Loss':<12} | {'Val Loss':<10} | {'Val Dice':<10} | {'Val IoU':<9} | {'Val Accuracy':<14} | {'Val Recall':<12}", flush=True)
    print("-" * 110, flush=True)

    for epoch in range(1, epochs + 1):
        start_t = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - start_t

        val_dice = val_metrics["val_dice"]
        val_loss = val_metrics["val_loss"]
        val_iou = val_metrics["val_iou"]
        val_acc = val_metrics["val_accuracy"]
        val_rec = val_metrics["val_recall"]
        val_prec = val_metrics["val_precision"]

        scheduler.step(val_dice)

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["val_accuracy"].append(val_acc)
        history["val_recall"].append(val_rec)
        history["val_precision"].append(val_prec)

        epoch_str = f"Epoch {epoch:02d}/{epochs:02d}"
        print(f"{epoch_str:<10} | {elapsed:>6.1f}s | {train_loss:>12.4f} | {val_loss:>10.4f} | {val_dice:>10.4f} | {val_iou:>9.4f} | {val_acc:>13.2f}% | {val_rec:>11.2f}%", flush=True)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                'model_state_dict': model.state_dict(),
                'init_filters': best_config['init_filters'],
                'dropout_rate': best_config['dropout_rate'],
                'input_shape': (256, 256, 1),
                'best_val_dice': best_val_dice,
                'hyperparams': best_config
            }, best_model_path)
            print(f"   ↳ [CHECKPOINT] Saved new best weights (Val Dice: {best_val_dice:.4f}, Accuracy: {val_acc:.2f}%, Recall: {val_rec:.2f}%)", flush=True)

    print("=" * 110 + "\n", flush=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["epoch"], history["train_loss"], label="Train Loss", color="#EF4444", linewidth=2)
    plt.plot(history["epoch"], history["val_loss"], label="Val Loss", color="#3B82F6", linewidth=2)
    plt.title("U-Net Loss Curve", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("BCE-Dice Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["epoch"], history["val_dice"], label="Val Dice Score", color="#10B981", linewidth=2)
    plt.plot(history["epoch"], history["val_accuracy"], label="Val Accuracy %", color="#8B5CF6", linewidth=2)
    plt.plot(history["epoch"], history["val_recall"], label="Val Recall %", color="#F59E0B", linewidth=2)
    plt.title("U-Net Performance Metrics", fontsize=12, fontweight='bold')
    plt.xlabel("Epoch")
    plt.ylabel("Score / Percentage")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    curve_path = os.path.join(reports_dir, "training_curves.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()

    print(f"[SUCCESS] Model training complete! Saved best model weights to: {best_model_path}", flush=True)
    print(f"[SUCCESS] Saved training curves plot to: {curve_path}\n", flush=True)

    print("=" * 80, flush=True)
    print("AUTOMATIC TEST SPLIT EVALUATION BREAKDOWN", flush=True)
    print("=" * 80, flush=True)
    try:
        from model.evaluate_test import evaluate_test_set
        evaluate_test_set(model_path=best_model_path, dataset_dir=dataset_dir, reports_dir=reports_dir, device=device)
    except Exception as e:
        print(f"[WARNING] Test evaluation failed: {e}", flush=True)

    return best_model_path, history


def main():
    try:
        parser = argparse.ArgumentParser(description="InSight3D - Train U-Net Model")
        parser.add_argument("--epochs", type=int, default=15, help="Number of additional training epochs (e.g. 15, 25, 50)")
        parser.add_argument("--resume", action="store_true", help="Resume/continue training from existing models/best_model.pth checkpoint")
        parser.add_argument("--skip-tuning", action="store_true", help="Skip hyperparameter grid search")
        args = parser.parse_args()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Using Hardware Acceleration Device: {device.upper()}", flush=True)

        dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
        best_model_file = os.path.join(PROJECT_ROOT, "models", "best_model.pth")

        if args.resume and os.path.exists(best_model_file):
            print(f"[INFO] Resuming training mode activated from checkpoint: {best_model_file}", flush=True)
            checkpoint = torch.load(best_model_file, map_location=device)
            best_config = checkpoint.get("hyperparams", {"lr": 0.0005, "init_filters": 32, "batch_size": 16, "dropout_rate": 0.2})
            train_ds = FastCTDefectDataset(os.path.join(dataset_dir, "train"))
            val_ds = FastCTDefectDataset(os.path.join(dataset_dir, "val"))
        elif args.skip_tuning:
            best_config = {"lr": 0.0005, "init_filters": 32, "batch_size": 16, "dropout_rate": 0.2}
            train_ds = FastCTDefectDataset(os.path.join(dataset_dir, "train"))
            val_ds = FastCTDefectDataset(os.path.join(dataset_dir, "val"))
        else:
            best_config, tuning_results, train_ds, val_ds = run_hyperparameter_tuning(dataset_dir=dataset_dir, device=device)

        best_model_path, history = train_best_model(
            best_config, train_ds, val_ds, output_model_dir=os.path.join(PROJECT_ROOT, "models"), reports_dir=os.path.join(PROJECT_ROOT, "results", "reports"), epochs=args.epochs, resume=args.resume, device=device
        )
    except Exception:
        traceback.print_exc(file=sys.stdout)
        sys.exit(1)


if __name__ == "__main__":
    main()
