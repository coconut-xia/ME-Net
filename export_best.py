"""Export MultiEdgeNet_CGF_EEF_2timesAMA best.pt to ONNX."""
import os
import sys
import torch
import torch.nn as nn

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ultralytics import YOLO

WEIGHTS = os.path.join(REPO_ROOT, "weights/me-net.pt")

model = YOLO(WEIGHTS)
model.model = model.model.float()

# Legacy checkpoint fix: add missing gate_scale to ACF
for m in model.model.modules():
    if hasattr(m, 'gate') and not hasattr(m, 'gate_scale'):
        m.gate_scale = nn.Parameter(torch.tensor(0.5))
        print(f"[FIX] Added gate_scale to ACF")

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
