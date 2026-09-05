"""
InSight3D Launcher Script
Executes Streamlit App from project subfolder.
"""
import subprocess
import sys
import os

if __name__ == "__main__":
    proj_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3D_Print_Defect_Detection")
    app_path = os.path.join(proj_dir, "app", "streamlit_app.py")
    print(f"Launching InSight3D Streamlit Dashboard ({app_path})...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], cwd=proj_dir)
