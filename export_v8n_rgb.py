"""Export v8n_RGB best.pt to ONNX."""
import os
import sys
import torch

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ultralytics import YOLO

WEIGHTS = os.path.join(REPO_ROOT, "weights", "v8n_RGB.pt")

model = YOLO(WEIGHTS)
model.model = model.model.float()

print(f"[INFO] Loaded model: {WEIGHTS}")

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
