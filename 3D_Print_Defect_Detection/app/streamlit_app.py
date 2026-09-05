"""
InSight3D - Interactive Streamlit Web Application
AI-Based Internal Defect Segmentation in 3D-Printed Electronic Components Using U-Net

Features:
- Live CT Slice Defect Segmentation & Quality Evaluation
- Dataset Selection from Test Split (63 real CT slices) or File Upload
- Real-time Preprocessing, U-Net Inference, Binary Thresholding & Color Overlay
- Automated PASS / WARNING / FAIL Quality Decision Engine
- Model Performance & Hyperparameter Tuning Dashboard
- PDF & CSV Inspection Report Downloads
"""

import os
import sys
import glob
import json
import numpy as np
import cv2
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure parent path is in sys.path and resolve absolute project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def get_path(*parts) -> str:
    """Helper to resolve paths relative to PROJECT_ROOT or current working dir."""
    path_root = os.path.join(PROJECT_ROOT, *parts)
    if os.path.exists(path_root):
        return path_root
    path_cwd = os.path.join(os.getcwd(), *parts)
    return path_cwd

from preprocessing.preprocess import preprocess_pipeline
from model.predict import DefectPredictor
from postprocessing.analyze import analyze_defects, QualityDecisionEngine
from postprocessing.report import generate_inspection_report
from utils.visualization import (
    plot_comparison,
    plot_probability_heatmap,
    plot_defect_contours
)

# Page Configuration
st.set_page_config(
    page_title="InSight3D - AI Defect Inspection Platform",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling based on Corporate Blue Color Palette
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
    }

    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #1C55A4 0%, #2563EB 50%, #87C4EC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }

    .sub-header {
        font-size: 1.05rem;
        font-weight: 500;
        color: #3A4750;
        margin-bottom: 2.0rem;
    }

    /* Card Containers */
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 14px rgba(11, 19, 43, 0.04);
        text-align: center;
        transition: all 0.2s ease-in-out;
    }

    .metric-card:hover {
        border-color: #87C4EC;
        box-shadow: 0 6px 18px rgba(28, 85, 164, 0.1);
    }

    .metric-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #475569;
        margin-bottom: 6px;
    }

    .metric-value {
        font-size: 1.5rem;
        font-weight: 800;
        color: #0B132B;
    }

    /* Quality Decision Badges */
    .status-badge {
        display: block;
        width: 100%;
        padding: 16px 20px;
        border-radius: 12px;
        font-weight: 800;
        font-size: 1.35rem;
        text-align: center;
        letter-spacing: 0.04em;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
    }

    .status-pass {
        background-color: #ECFDF5;
        color: #059669;
        border: 2px solid #10B981;
    }

    .status-warning {
        background-color: #FFFBEB;
        color: #D97706;
        border: 2px solid #F59E0B;
    }

    .status-fail {
        background-color: #FDF2F4;
        color: #722F37;
        border: 2px solid #800020;
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #0B132B !important;
    }

    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown {
        color: #F1F5F8 !important;
        font-weight: 600 !important;
    }

    /* Streamlit Primary Accent Color & Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 2px solid #E2E8F0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 8px 8px 0px 0px;
        font-weight: 700;
        color: #475569;
        padding: 0px 20px;
    }

    .stTabs [aria-selected="true"] {
        color: #1C55A4 !important;
    /* Complete Overrides for Streamlit Red UI Elements to Wine Red (#722F37) */

    /* 1. Checkboxes */
    div[data-baseweb="checkbox"] div[aria-checked="true"] {
        background-color: #722F37 !important;
        border-color: #722F37 !important;
    }
    div[data-baseweb="checkbox"] input:checked + div {
        background-color: #722F37 !important;
        border-color: #722F37 !important;
    }

    /* 2. Sliders (Knobs, Tracks, and Value Numbers) */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #722F37 !important;
        border-color: #722F37 !important;
    }
    .stSlider [data-baseweb="slider"] div[style*="background-color"] {
        background-color: #722F37 !important;
    }
    .stSlider [data-baseweb="slider"] [aria-valuenow] {
        color: #722F37 !important;
    }
    .stSlider [data-testid="stWidgetLabel"] + div p {
        color: #722F37 !important;
        font-weight: 700 !important;
    }

    /* 3. Radio Buttons */
    div[data-baseweb="radio"] div[aria-checked="true"] {
        background-color: #722F37 !important;
        border-color: #722F37 !important;
    }
    div[data-baseweb="radio"] div[aria-checked="true"] div {
        background-color: #722F37 !important;
    }

    /* 4. Tab Underline Highlight Bar */
    div[data-baseweb="tab-highlight"] {
        background-color: #722F37 !important;
    }

    /* Button Styling */
    div.stDownloadButton > button, .stButton > button {
        background: linear-gradient(135deg, #1C55A4 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 12px rgba(28, 85, 164, 0.25) !important;
        transition: all 0.2s ease-in-out !important;
        width: 100% !important;
    }

    div.stDownloadButton > button:hover, .stButton > button:hover {
        background: linear-gradient(135deg, #1E40AF 0%, #1C55A4 100%) !important;
        box-shadow: 0 6px 18px rgba(28, 85, 164, 0.4) !important;
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_predictor(model_path: str):
    return DefectPredictor(model_path=model_path, img_size=(256, 256))


def main():
    st.markdown('<div class="main-header">InSight3D Internal Defect Inspection Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Automated Internal Defect Segmentation in 3D-Printed Electronic Components Using U-Net</div>', unsafe_allow_html=True)

    # Sidebar Controls
    st.sidebar.title("Inspection Controls")
    
    model_path = get_path("models", "best_model.pth")
    predictor = load_predictor(model_path)

    st.sidebar.markdown("---")
    st.sidebar.subheader("1. Preprocessing Configuration")
    use_clahe = st.sidebar.checkbox("Apply CLAHE Contrast Enhancement", value=True)
    clahe_clip = st.sidebar.slider("CLAHE Clip Limit", 0.5, 5.0, 2.0, 0.5)
    noise_method = st.sidebar.selectbox("Noise Reduction Filter", ["gaussian", "median", "none"], index=0)
    kernel_size = st.sidebar.slider("Filter Kernel Size", 3, 9, 3, step=2)

    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Segmentation Threshold")
    pred_threshold = st.sidebar.slider("Defect Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Quality Decision Rules")
    pass_thresh = st.sidebar.number_input("Pass Defect Area Limit (%)", value=1.0, step=0.5)
    warn_thresh = st.sidebar.number_input("Warning Defect Area Limit (%)", value=3.0, step=0.5)

    component_id = st.sidebar.text_input("Component Serial ID", value="COMP-3D-ELEX-901")

    # Main Navigation Tabs
    tab_inspection, tab_performance = st.tabs([
        "🔍 CT Slice Defect Segmentation",
        "📈 Model Performance & Tuning Results"
    ])

    # =========================================================================
    # TAB 1: LIVE DEFECT SEGMENTATION & INSPECTION
    # =========================================================================
    with tab_inspection:
        col_input_type, col_select = st.columns([1, 2])
        
        with col_input_type:
            input_source = st.radio(
                "Select CT Image Source:",
                ["Dataset Test Split (63 Slices)", "Upload Custom Image"],
                index=0
            )

        input_img = None
        input_filename = ""

        if input_source == "Dataset Test Split (63 Slices)":
            test_img_dir = get_path("dataset", "test", "images")
            test_img_paths = sorted(glob.glob(os.path.join(test_img_dir, "*.png")) + glob.glob(os.path.join(test_img_dir, "*.jpg")))
            
            if test_img_paths:
                with col_select:
                    selected_file = st.selectbox("Select Test CT Slice:", test_img_paths, format_func=lambda x: os.path.basename(x))
                input_img = cv2.imread(selected_file, cv2.IMREAD_GRAYSCALE)
                input_filename = os.path.basename(selected_file)
            else:
                st.warning("No test images found in dataset/test/images directory.")
        else:
            with col_select:
                uploaded_file = st.file_uploader("Upload X-Ray CT Slice Image", type=["png", "jpg", "jpeg", "tif", "bmp"])
            if uploaded_file is not None:
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                input_img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
                input_filename = uploaded_file.name

        if input_img is None:
            st.info("Please select a CT slice image above to run defect inspection.")
        else:
            # Run Pipeline
            prep_img, prob_map, binary_mask = predictor.predict(
                input_img,
                use_clahe=use_clahe,
                noise_reduction=None if noise_method == "none" else noise_method,
                threshold=pred_threshold
            )
            overlay_img = predictor.create_overlay(prep_img, binary_mask, color=(32, 0, 128), alpha=0.5)

            rules_engine = QualityDecisionEngine(pass_threshold=pass_thresh, warning_threshold=warn_thresh)
            analysis_result = analyze_defects(binary_mask, decision_engine=rules_engine)

            # Results KPI Cards
            st.markdown("---")
            st.subheader("Quality Decision & Metric Summary")
            status = analysis_result["status"]
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)

            with kpi1:
                if status == "PASS":
                    st.markdown('<div class="status-badge status-pass">✅ PASS</div>', unsafe_allow_html=True)
                elif status == "WARNING":
                    st.markdown('<div class="status-badge status-warning">⚠️ WARNING</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-badge status-fail">❌ FAIL</div>', unsafe_allow_html=True)

            with kpi2:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-title">Defect Region Count</div>
                    <div class="metric-value">{analysis_result["defect_count"]} Voids</div>
                </div>
                ''', unsafe_allow_html=True)

            with kpi3:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-title">Defect Area Ratio</div>
                    <div class="metric-value">{analysis_result["defect_percentage"]}%</div>
                </div>
                ''', unsafe_allow_html=True)

            with kpi4:
                st.markdown(f'''
                <div class="metric-card">
                    <div class="metric-title">Largest Defect Area</div>
                    <div class="metric-value">{analysis_result["largest_defect"]} px²</div>
                </div>
                ''', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Visual Panels
            vtab1, vtab2, vtab3, vtab4 = st.tabs([
                "📷 3-Panel Inspection View",
                "🔥 Probability Heatmap",
                "🎯 Defect Contours & Indexing",
                "📊 Defect Inventory Table"
            ])

            with vtab1:
                fig_comp = plot_comparison(prep_img, binary_mask, overlay_img, title=f"InSight3D CT Slice View: {input_filename}")
                st.pyplot(fig_comp)
                plt.close(fig_comp)

            with vtab2:
                fig_heat = plot_probability_heatmap(prob_map, title="U-Net Pixel Defect Probability Heatmap")
                st.pyplot(fig_heat)
                plt.close(fig_heat)

            with vtab3:
                fig_cnt = plot_defect_contours(prep_img, analysis_result)
                st.pyplot(fig_cnt)
                plt.close(fig_cnt)

            with vtab4:
                defects_data = analysis_result.get("defects", [])
                if defects_data:
                    df_defects = pd.DataFrame(defects_data)
                    df_defects.columns = ["Defect ID", "Area (px²)", "Centroid (X, Y)", "Bounding Box [X, Y, W, H]"]
                    st.dataframe(df_defects, use_container_width=True)
                else:
                    st.info("No internal defects detected in component slice.")

            # Report Downloads
            st.markdown("---")
            reports_dir = get_path("results", "reports")
            reports_dict = generate_inspection_report(analysis_result, component_id=component_id, reports_dir=reports_dir)

            col_csv, col_pdf = st.columns(2)
            with col_csv:
                if os.path.exists(reports_dict["csv_report"]):
                    with open(reports_dict["csv_report"], "rb") as csv_f:
                        st.download_button("📥 Download CSV Log", csv_f, file_name=f"InSight3D_{component_id}.csv", mime="text/csv")
            with col_pdf:
                if os.path.exists(reports_dict["pdf_report"]):
                    with open(reports_dict["pdf_report"], "rb") as pdf_f:
                        st.download_button("📄 Download PDF Certificate", pdf_f, file_name=f"InSight3D_{component_id}.pdf", mime="application/pdf")

    # =========================================================================
    # TAB 2: MODEL PERFORMANCE & HYPERPARAMETER TUNING RESULTS
    # =========================================================================
    with tab_performance:
        st.subheader("U-Net Model Training & Validation Performance")

        perf_col1, perf_col2 = st.columns(2)

        with perf_col1:
            st.markdown("### 📊 Hyperparameter Tuning Grid Search Results")
            tuning_csv = get_path("results", "reports", "hyperparameter_tuning_results.csv")
            if os.path.exists(tuning_csv):
                df_tuning = pd.read_csv(tuning_csv)
                st.dataframe(df_tuning, use_container_width=True)
                
                # Chart
                st.bar_chart(df_tuning, x="trial_id", y=["val_dice", "val_iou"], use_container_width=True)
            else:
                st.info("Hyperparameter tuning report generating...")

        with perf_col2:
            st.markdown("### 📈 Training & Validation History Curves")
            curves_path = get_path("results", "reports", "training_curves.png")
            if os.path.exists(curves_path):
                st.image(curves_path, use_container_width=True)
            else:
                st.info("Training history curves generating...")

        st.markdown("---")
        st.subheader("🧪 Test Set Evaluation Report (Unseen Test Split)")
        
        test_col1, test_col2 = st.columns([1, 2])

        with test_col1:
            test_csv = get_path("results", "reports", "test_evaluation_report.csv")
            if os.path.exists(test_csv):
                df_test = pd.read_csv(test_csv)
                st.dataframe(df_test.T.rename(columns={0: "Metric Value"}), use_container_width=True)
            else:
                st.info("Test set evaluation report generating...")

        with test_col2:
            grid_img_path = get_path("results", "reports", "test_predictions_grid.png")
            if os.path.exists(grid_img_path):
                st.image(grid_img_path, caption="Test Set Prediction Overlays Grid", use_container_width=True)


if __name__ == "__main__":
    main()
