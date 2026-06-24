"""Train ME-Net on a paired RGB/Event YOLO dataset."""

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "ultralytics/cfg/models/ME-Net/ME-Net.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Dataset YAML")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="ME-Net YAML or checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device", default="", help="Ultralytics device string, e.g. 0,1 or cpu")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/train")
    parser.add_argument("--name", default="me-net")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model.resolve()))
    model.train(
        data=str(args.data.resolve()),
        ch=6,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        close_mosaic=15,
        workers=args.workers,
        device=args.device,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        weight_decay=0.0005,
        momentum=0.937,
        cos_lr=True,
        patience=20,
        amp=True,
        cache=False,
        project=str(args.project.resolve()),
        name=args.name,
        resume=False,
        deterministic=False,
        seed=0,
        warmup_epochs=5.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.3,
        degrees=10.0,
        translate=0.2,
        scale=0.4,
        shear=2.0,
        perspective=0.0,
        fliplr=0.5,
        mosaic=0.7,
        mixup=0.1,
        copy_paste=0.15,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        val=True,
        plots=True,
        save=True,
    )


if __name__ == "__main__":
    main()
