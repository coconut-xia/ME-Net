"""
Export ME-Net model to ONNX format for TensorRT conversion on Jetson Orin.

Usage:
    python export_onnx.py

After export, copy the .onnx file to Jetson Orin and run:
    /usr/src/tensorrt/bin/trtexec \\
        --onnx=best.onnx \\
        --saveEngine=best_fp16.engine \\
        --fp16 \\
        --workspace=4096
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from ultralytics import YOLO

WEIGHTS = os.path.join(REPO_ROOT, "weights", "me-net.pt")

model = YOLO(WEIGHTS)

# Convert FP16 weights to FP32 for export compatibility
model.model = model.model.float()

print(f"[INFO] Loaded model: {WEIGHTS}")
print(f"[INFO] Input channels: 6 (RGB 3ch + Events 3ch), imgsz: 640")

model.export(
    format="onnx",
    imgsz=640,
    batch=1,
    half=False,
    simplify=True,
    opset=17,
    dynamic=False,
    device="cpu",
)

print("[INFO] ONNX export complete.")
print("[INFO] Next: copy the .onnx file to Jetson Orin, then run:")
print()
print("  # FP16 (recommended, fastest)")
print("  /usr/src/tensorrt/bin/trtexec \\")
print("      --onnx=best.onnx \\")
print("      --saveEngine=best_fp16.engine \\")
print("      --fp16 \\")
print("      --workspace=4096")
print()
print("  # Or use ultralytics (requires ultralytics on Jetson)")
print("  yolo export model=best.onnx format=engine half=True device=0")
