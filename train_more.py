"""
InSight3D Root Launcher - Continue Model Training (Unbuffered Live Stream)
Usage:
  python train_more.py --epochs 25
"""

import os
import sys
import subprocess

if __name__ == "__main__":
    proj_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3D_Print_Defect_Detection")
    script_path = os.path.join(proj_dir, "model", "tune_and_train.py")
    cmd = [sys.executable, "-u", script_path, "--resume"] + sys.argv[1:]
    print(f"Executing Live Training: {' '.join(cmd)}\n", flush=True)
    subprocess.run(cmd, cwd=proj_dir)
