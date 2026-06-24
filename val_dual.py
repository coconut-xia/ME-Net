"""Validate an ME-Net checkpoint on a paired RGB/Event YOLO dataset."""

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "weights/me-net.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Dataset YAML")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/val")
    parser.add_argument("--name", default="eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.weights.resolve()))
    model.val(
        data=str(args.data.resolve()),
        ch=6,
        imgsz=args.imgsz,
        workers=args.workers,
        device=args.device,
        batch=args.batch,
        project=str(args.project.resolve()),
        name=args.name,
    )


if __name__ == "__main__":
    main()
