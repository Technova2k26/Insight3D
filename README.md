# InSight3D — AI-Based Internal Defect Segmentation in 3D-Printed Components

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://insight3d-mkde569qrch54c5df5ub3c.streamlit.app)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/Technova2k26/Insight3D)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)

**InSight3D** is a non-destructive evaluation (NDE) software platform for detecting and segmenting internal defects (micro-voids, gas porosity, layer delamination) in 3D-printed electronic components from X-ray CT slice images using a custom PyTorch U-Net model.

---

## 🌐 Live Web Application

Access the live cloud-deployed Streamlit web app:  
👉 **[https://insight3d-mkde569qrch54c5df5ub3c.streamlit.app](https://insight3d-mkde569qrch54c5df5ub3c.streamlit.app)**

GitHub Repository:  
👉 **[https://github.com/Technova2k26/Insight3D](https://github.com/Technova2k26/Insight3D)**

---

## Key Features

- **Non-Destructive X-Ray Inspection**: Analyzes internal CT slice scans ($256 \times 256$) without physically sectioning or damaging components.
- **Deep Learning Segmentation**: 4-stage PyTorch U-Net architecture trained with a combined BCE-Dice Loss function to handle class imbalance (voids occupying <3% of image area).
- **Quantitative Defect Analytics**: Computes total defect count, individual void area ($\text{px}^2$), centroids $(X, Y)$, bounding boxes, and defect area percentages.
- **Automated Quality Grading**: Evaluates structural integrity into **PASS (<1.0% defect area)**, **WARNING (1.0%–3.0%)**, or **FAIL (>3.0%)** categories.
- **Interactive UI & Export**: Streamlit dashboard with continuous probability heatmaps, contour indexing, and one-click export for **PDF Quality Certificates** and **CSV logs**.

---

## System Performance & Benchmarks

Evaluated on 63 unseen test CT slices (`dataset/test`):

| Metric | Score / Value | Description |
| :--- | :--- | :--- |
| **Pixel Accuracy** | **96.96%** | Correct classification across background and solid material |
| **Dice Coefficient** | **0.4753** | Spatial overlap agreement (+197% improvement over baseline) |
| **Defect Recall** | **50.88%** | Sensitivity in identifying sub-millimeter internal voids |
| **Precision** | **44.61%** | False-positive noise suppression |
| **Inference Speed** | **~411 ms / slice** | Real-time execution on standard CPU hardware |

---

## Project Structure

```text
Insight3D/
├── app.py                     # Root entrypoint for Streamlit deployment
├── run_app.py                 # Local application launcher
├── train_more.py              # Root script to run additional training epochs
├── requirements.txt           # Python dependencies
├── .gitignore                 # Tracked file exclusions
├── .streamlit/                # Custom UI color themes and settings
└── 3D_Print_Defect_Detection/
    ├── app/
    │   └── streamlit_app.py   # Main Streamlit web application interface
    ├── dataset/               # Preprocessed dataset splits (train: 288, val: 61, test: 63)
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── model/
    │   ├── unet_torch.py      # PyTorch 4-stage U-Net model definition
    │   ├── tune_and_train.py  # Grid search tuning and training engine
    │   ├── evaluate_test.py   # Benchmark evaluation script for test split
    │   └── predict.py         # Prediction pipeline and overlay generator
    ├── models/
    │   └── best_model.pth     # Trained PyTorch model weights
    ├── postprocessing/
    │   ├── analyze.py         # OpenCV defect contour tracking & Quality Decision Engine
    │   └── report.py          # ReportLab PDF certificate & CSV report generator
    ├── preprocessing/
    │   ├── preprocess.py      # CLAHE contrast enhancement & noise reduction
    │   └── prepare_dataset.py # Automated dataset splitter (70/15/15)
    ├── results/
    │   └── reports/           # Training curves, tuning CSVs, and PDF outputs
    └── utils/
        ├── metrics.py         # Dice coefficient, IoU, and loss functions
        └── visualization.py   # Matplotlib inspection visualizers & heatmaps
```

---

## Quick Start & Local Setup

### 1. Prerequisites
Ensure you have **Python 3.10+** and **Git** installed on your system.

### 2. Clone Repository
```bash
git clone https://github.com/Technova2k26/Insight3D.git
cd Insight3D
```

### 3. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## How to Run

### Launch Web Dashboard
To start the interactive Streamlit inspection interface locally:
```bash
python run_app.py
```
Or run directly via Streamlit:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Run Additional Model Training Epochs
To continue training the U-Net model from existing checkpoint weights:
```bash
python train_more.py --epochs 15
```

### Evaluate Model on Test Dataset
To run benchmark evaluations on the 63 unseen test slices:
```bash
python 3D_Print_Defect_Detection/model/evaluate_test.py
```

---

## Technology Stack

- **Primary Language**: Python 3.10+
- **Deep Learning**: PyTorch (TorchVision, Torch)
- **Computer Vision**: OpenCV (`opencv-python-headless`)
- **Data & Numerical Processing**: NumPy, Pandas
- **Visualization & UI**: Streamlit, Matplotlib
- **Reporting Engine**: ReportLab (PDF Certificates)

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
