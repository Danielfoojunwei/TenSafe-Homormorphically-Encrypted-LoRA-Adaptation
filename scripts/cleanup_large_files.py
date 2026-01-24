
import os
from pathlib import Path

files_to_delete = [
    "TensorGuardFlow_Production_Ready.zip",
    "TensorGuardFlow_Hardened.zip",
    "data/fastumi_pro/task1/session_1/left_hand_250801DR48FP25002314/RGB_Images/video.mp4",
    "data/synthetic/video.mp4"
]

root = Path(".")
for f in files_to_delete:
    p = root / f
    if p.exists():
        try:
            p.unlink()
            print(f"Deleted: {p}")
        except Exception as e:
            print(f"Failed to delete {p}: {e}")
    else:
        print(f"Not found: {p}")
