# InSight3D - AI-Based Internal Defect Segmentation in 3D-Printed Electronic Components Using U-Net

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

**InSight3D** is a production-grade AI defect inspection platform designed to automatically segment and analyze internal structural defects in 3D-printed electronic components from X-ray Computed Tomography (CT) slice images.

3D-printed electronic components often suffer from non-destructive internal flaws such as:
- 🔴 **Voids**: Trapped air or gas pockets inside printed substrate layers.
- 🟡 **Porosity**: Micro-void clusters reducing dielectric and structural strength.
- 🟠 **Delamination**: Inter-layer separation between printed conductive trace/dielectric materials.
- 🟣 **Unfused Powder**: Unbonded raw metal/polymer particles inside internal cavities.

Using a custom **U-Net** deep learning architecture built from scratch in TensorFlow/Keras, **InSight3D** isolates defect regions at pixel-level resolution, measures key geometric metrics (area, bounding box, centroid, defect percentage), and assigns an automated Quality Status (**PASS / WARNING / FAIL**).

---

## 📁 Repository Structure

```
3D_Print_Defect_Detection/
├── dataset/                    # Dataset directory (Populated by teammate)
│   ├── train/                  # Training set split
│   │   ├── images/             # Raw CT slice images (*.png, *.jpg, *.tif)
│   │   └── masks/              # Binary defect ground truth masks
│   ├── val/                    # Validation set split
│   │   ├── images/
│   │   └── masks/
│   └── test/                   # Test evaluation split
│       ├── images/
│       └── masks/
├── preprocessing/              # Step 1 & 2: Preprocessing & Data Augmentation
│   ├── __init__.py
│   ├── preprocess.py           # Resizing, Normalization, CLAHE, Noise Removal
│   └── augment.py              # Geometric & intensity augmentations (Flip, Rotate, Zoom)
├── model/                      # Step 3, 5 & 6: U-Net Architecture & Pipelines
│   ├── __init__.py
│   ├── unet.py                 # Full U-Net architecture built from scratch (Encoder, Bottleneck, Decoder)
│   ├── train.py                # Training pipeline with Callbacks, TensorBoard, and checkpointing
│   └── predict.py              # Inference pipeline, thresholding, and overlay creation
├── postprocessing/             # Step 7, 8 & 11: Defect Analysis & Reports
│   ├── __init__.py
│   ├── analyze.py              # Contour extraction, area calculation, Quality Decision Engine
│   └── report.py               # Automated PDF & CSV Inspection Report generator
├── utils/                      # Step 4 & 9: Custom Metrics & Visualization
│   ├── __init__.py
│   ├── metrics.py              # Dice Coef, Dice Loss, IoU, Precision, Recall, BCE-Dice Loss
│   └── visualization.py        # Matplotlib visualizers (3-panel view, heatmaps, contours, dashboard)
├── app/                        # Step 10: Streamlit Dashboard App
│   └── streamlit_app.py        # Interactive web UI with upload, live inference, and PDF downloads
├── results/                    # Output directory for saved artifacts
│   ├── overlays/               # Saved defect highlight overlays
│   ├── masks/                  # Saved predicted binary masks
│   └── reports/                # Exported PDF and CSV inspection certificates
├── requirements.txt            # Locked package dependencies
└── README.md                   # Project documentation
```

---

## ⚡ Quick Start & Setup

### 1. Installation

Clone the repository and install required dependencies:
```bash
cd 3D_Print_Defect_Detection
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application

To launch the interactive inspection dashboard:
```bash
streamlit run app/streamlit_app.py
```
> 💡 **Note**: The app defaults to **Synthetic Demo Mode** if no real dataset is loaded yet, allowing instant verification of the complete UI, probability heatmaps, contour indexing, and PDF report downloads!

---

## 🔄 Dataset Integration Guide for Teammates

If you are uploading collected X-ray CT slice images, place your files according to the following convention:

1. **Format Requirements**:
   - Grayscale CT images: `.png`, `.jpg`, `.jpeg`, or `.tif`
   - Binary Ground Truth Defect Masks: Same filename as corresponding image, where white pixels (`255`) represent defects and black (`0`) represents clean substrate.
2. **Directory Placement**:
   ```
   dataset/train/images/slice_001.png
   dataset/train/masks/slice_001.png

   dataset/val/images/slice_050.png
   dataset/val/masks/slice_050.png
   ```

### Training U-Net Model on Collected Dataset

Once files are placed in `dataset/train` and `dataset/val`, start training via:
```bash
python model/train.py --epochs 25 --batch-size 8 --lr 0.0001
```
The trained model weights will automatically be saved to `models/best_model.keras`.

---

## 🔬 Pipeline Workflow Summary

```
Raw CT Image 
    │
    ▼
Preprocessing (Resize 256x256 ➔ CLAHE Contrast ➔ Gaussian Noise Filter ➔ Normalize [0,1])
    │
    ▼
U-Net Deep Model Inference ➔ Pixel Defect Probability Map
    │
    ▼
Thresholding (>=0.5) ➔ Binary Defect Mask ➔ Color Overlay Creation
    │
    ▼
OpenCV Contour Extraction ➔ Metrics (Defect Count, Area px², Centroid, Defect %)
    │
    ▼
Quality Decision Engine Rules (<1% PASS, 1-3% WARNING, >3% FAIL)
    │
    ▼
Inspection Reports Exported (CSV & PDF Formats)
```

---

## 🛠️ Tech Stack

- **Core Logic & Computer Vision**: Python 3.9+, OpenCV, NumPy, SciPy
- **Deep Learning**: TensorFlow / Keras 3 (Custom U-Net Architecture, Custom BCE-Dice Loss)
- **Data Analysis & Visualization**: Pandas, Matplotlib, ReportLab (PDF Generation)
- **Web Application**: Streamlit

---

## 🔮 Future Roadmap

- [ ] Support for 3D volumetric DICOM stack segmentation (.dcm / .nii).
- [ ] Multi-class segmentation distinguishing specific defect types (Void vs. Delamination vs. Porosity).
- [ ] Edge-AI optimization (TensorRT / ONNX export) for real-time in-line 3D printer quality control.
