"""
InSight3D - Visualization Module
AI-Based Internal Defect Segmentation in 3D-Printed Electronic Components Using U-Net

Functions using Matplotlib & OpenCV to render:
1. Side-by-Side Comparison (Original CT, Binary Mask, Overlay)
2. Defect Contours Plot with Indexing
3. Defect Probability Heatmap
4. Inspection Executive Dashboard
"""

import os
from typing import Dict, Any, Optional, Tuple
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless execution and web UI rendering
import matplotlib.pyplot as plt


def plot_comparison(
    original_img: np.ndarray,
    predicted_mask: np.ndarray,
    overlay_img: np.ndarray,
    title: str = "InSight3D Defect Inspection Comparison",
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot 3-panel comparison: Original CT slice, Predicted defect mask, and Defect overlay.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    # Panel 1: Original CT
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title("1. Original CT Slice", fontsize=12)
    axes[0].axis('off')

    # Panel 2: Predicted Mask
    axes[1].imshow(predicted_mask, cmap='inferno')
    axes[1].set_title("2. Predicted Defect Mask", fontsize=12)
    axes[1].axis('off')

    # Panel 3: Overlay
    if len(overlay_img.shape) == 3 and overlay_img.shape[2] == 3:
        # Convert BGR to RGB for Matplotlib
        overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
    else:
        overlay_rgb = overlay_img
    axes[2].imshow(overlay_rgb)
    axes[2].set_title("3. Defect Highlight Overlay", fontsize=12)
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


def plot_probability_heatmap(
    prob_map: np.ndarray,
    title: str = "Defect Probability Map",
    figsize: Tuple[int, int] = (6, 5),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot continuous defect probability heatmap with colorbar.
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(prob_map, cmap='plasma', vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axis('off')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Defect Probability', rotation=270, labelpad=15)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


def plot_defect_contours(
    original_img: np.ndarray,
    defects_info: Dict[str, Any],
    figsize: Tuple[int, int] = (6, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot CT slice image with defect bounding boxes, centroids, and IDs overlayed.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(original_img, cmap='gray')
    ax.set_title(f"Defect Contours & Indexing (Total: {defects_info.get('defect_count', 0)})", fontsize=12, fontweight='bold')

    for defect in defects_info.get("defects", []):
        x, y, w, h = defect["bbox"]
        cx, cy = defect["centroid"]
        d_id = defect["id"]

        # Draw Bounding Box (Wine Red)
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='#800020', linewidth=1.5)
        ax.add_patch(rect)

        # Plot Centroid Point
        ax.plot(cx, cy, 'r+', markersize=8)

        # ID Annotation
        ax.text(x, max(0, y - 4), f"#{d_id} ({defect['area']:.0f}px)", color='yellow', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='square,pad=0.2', facecolor='black', alpha=0.6))

    ax.axis('off')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


def plot_dashboard(
    original_img: np.ndarray,
    predicted_mask: np.ndarray,
    prob_map: np.ndarray,
    overlay_img: np.ndarray,
    analysis_result: Dict[str, Any],
    figsize: Tuple[int, int] = (12, 10),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Comprehensive 2x2 Quality Control Dashboard.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    status = analysis_result.get("status", "PASS")
    status_color = "green" if status == "PASS" else ("orange" if status == "WARNING" else "red")

    fig.suptitle(
        f"InSight3D Inspection Dashboard - Status: {status} (Defects: {analysis_result.get('defect_count')}, Defect Area: {analysis_result.get('defect_percentage')}%)",
        fontsize=14, fontweight='bold', color=status_color, y=0.98
    )

    # Panel 1: Original CT Slice
    axes[0, 0].imshow(original_img, cmap='gray')
    axes[0, 0].set_title("Original CT Slice", fontsize=11)
    axes[0, 0].axis('off')

    # Panel 2: Defect Probability Heatmap
    im2 = axes[0, 1].imshow(prob_map, cmap='plasma', vmin=0.0, vmax=1.0)
    axes[0, 1].set_title("Probability Heatmap", fontsize=11)
    axes[0, 1].axis('off')
    fig.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Panel 3: Binary Defect Mask
    axes[1, 0].imshow(predicted_mask, cmap='inferno')
    axes[1, 0].set_title("Binary Defect Mask", fontsize=11)
    axes[1, 0].axis('off')

    # Panel 4: Color Overlay
    overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB) if len(overlay_img.shape) == 3 else overlay_img
    axes[1, 1].imshow(overlay_rgb)
    axes[1, 1].set_title("Color Defect Overlay", fontsize=11)
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)

    return fig


if __name__ == "__main__":
    print("InSight3D Visualization Module Test")
    
    orig = np.random.randint(50, 200, size=(256, 256), dtype=np.uint8)
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[100:130, 100:130] = 1

    overlay = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)
    overlay[mask == 1] = [0, 0, 255]

    fig = plot_comparison(orig, mask, overlay, save_path="results/test_viz.png")
    print(f"Saved test visualization to results/test_viz.png")
