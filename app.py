"""
InSight3D Root App Entrypoint for Hugging Face Spaces Deployment
"""
import os
import sys

# Resolve absolute paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.join(ROOT_DIR, "3D_Print_Defect_Detection")

sys.path.append(ROOT_DIR)
sys.path.append(PROJ_DIR)
sys.path.append(os.path.join(PROJ_DIR, "app"))

from streamlit_app import main

if __name__ == "__main__":
    main()
