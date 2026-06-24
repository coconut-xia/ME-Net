"""
Predict on a sequence of paired RGB + DVS images using a dual-modal (ch=6) model.
Saves annotated RGB and DVS images separately.

Usage:
    python predict_sequence.py
"""

import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ultralytics import YOLO
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from ultralytics.data.augment import LetterBox

# ── Configuration ──────────────────────────────────────────
WEIGHT = os.path.join(REPO_ROOT, "weights", "me-net.pt")
DATA_ROOT = os.path.join(REPO_ROOT, "data", "sequences")
OUT_ROOT = os.path.join(REPO_ROOT, "outputs")

CONF = 0.25
IOU = 0.45
IMGSZ = 640
DEVICE = "cuda:0"

CLASS_NAMES = ["DJI M350RTK", "DJI Mavic 4 Pro", "DJI Air3", "DJI Avata2"]
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]
# ───────────────────────────────────────────────────────────


def letterbox_img(img: np.ndarray, new_shape=640):
    """Resize + pad to square, return (resized_img, ratio, (dw, dh))."""
    lb = LetterBox(new_shape, auto=False, stride=32)
    return lb(image=img)


def preprocess(rgb: np.ndarray, dvs: np.ndarray, device, imgsz=640):
    """Stack rgb+dvs → (1, 6, H, W) float32 tensor on device."""
    lb = LetterBox(imgsz, auto=False, stride=32)
    rgb_lb = lb(image=rgb)
    dvs_lb = lb(image=dvs)

    merged = np.concatenate([dvs_lb, rgb_lb], axis=2)
    merged = merged[..., ::-1]
    merged = merged.transpose(2, 0, 1)
    merged = np.ascontiguousarray(merged, dtype=np.float32) / 255.0
    return torch.from_numpy(merged).unsqueeze(0).to(device)


def draw_boxes(img: np.ndarray, boxes, confs, cls_ids) -> np.ndarray:
    out = img.copy()
    for box, conf, cid in zip(boxes, confs, cls_ids):
        x1, y1, x2, y2 = map(int, box)
        cid = int(cid)
        color = COLORS[cid % len(COLORS)]
        label = f"{CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else cid} {conf:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def sorted_images(folder: str):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [f for f in Path(folder).iterdir() if f.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.name)


def main():
    rgb_dir = os.path.join(DATA_ROOT, "rgb_frame")
    dvs_dir = os.path.join(DATA_ROOT, "dvs_frame")
    out_rgb = os.path.join(OUT_ROOT, "rgb_pred")
    out_dvs = os.path.join(OUT_ROOT, "dvs_pred")
    os.makedirs(out_rgb, exist_ok=True)
    os.makedirs(out_dvs, exist_ok=True)

    rgb_files = sorted_images(rgb_dir)
    dvs_map = {f.name: f for f in sorted_images(dvs_dir)}

    print(f"Loading model: {WEIGHT}")
    yolo = YOLO(WEIGHT)
    model = yolo.model.to(DEVICE).eval()
    nc = len(yolo.names)

    print(f"Found {len(rgb_files)} rgb frames, running prediction...")

    with torch.no_grad():
        for rgb_path in rgb_files:
            fname = rgb_path.name

            rgb_img = cv2.imread(str(rgb_path))
            dvs_path = dvs_map.get(fname)
            dvs_img = cv2.imread(str(dvs_path)) if dvs_path else None

            if rgb_img is None:
                print(f"[WARN] Cannot read {rgb_path}, skip.")
                continue

            if dvs_img is None:
                dvs_img = rgb_img.copy()

            im_tensor = preprocess(rgb_img, dvs_img, DEVICE, IMGSZ)

            preds = model(im_tensor)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]

            det = non_max_suppression(preds, CONF, IOU, nc=nc)[0]

            if det is not None and len(det):
                det[:, :4] = scale_boxes(im_tensor.shape[2:], det[:, :4], rgb_img.shape)
                boxes = det[:, :4].cpu().numpy()
                confs = det[:, 4].cpu().numpy()
                cls_ids = det[:, 5].cpu().numpy()
            else:
                boxes = confs = cls_ids = []

            rgb_drawn = draw_boxes(rgb_img, boxes, confs, cls_ids)
            cv2.imwrite(os.path.join(out_rgb, fname), rgb_drawn)

            if dvs_path is not None:
                dvs_drawn = draw_boxes(dvs_img, boxes, confs, cls_ids)
                cv2.imwrite(os.path.join(out_dvs, fname), dvs_drawn)

    print(f"\nDone.\n  rgb: {out_rgb}\n  dvs: {out_dvs}")


if __name__ == "__main__":
    main()
