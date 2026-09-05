# InSight3D - AI-Based Internal Defect Segmentation in 3D-Printed Electronic Components Using U-Net

Welcome to **InSight3D**!

The project files are structured inside `3D_Print_Defect_Detection/`:

- **Streamlit Web Application**: `3D_Print_Defect_Detection/app/streamlit_app.py`
- **U-Net Architecture**: `3D_Print_Defect_Detection/model/unet.py`
- **Preprocessing Module**: `3D_Print_Defect_Detection/preprocessing/preprocess.py`
- **Training Pipeline**: `3D_Print_Defect_Detection/model/train.py`
- **Defect Analysis & Rules**: `3D_Print_Defect_Detection/postprocessing/analyze.py`
- **PDF/CSV Report Generator**: `3D_Print_Defect_Detection/postprocessing/report.py`

## Quick Launch

To run the Streamlit App:
```bash
streamlit run 3D_Print_Defect_Detection/app/streamlit_app.py
```
Or execute:
```bash
python run_app.py
```
