# ME-Net

ME-Net is a six-channel RGB-DVS object detector built on Ultralytics YOLO. It uses dual RGB/event backbones with MER event refinement, ACF cross-modal fusion, and AMA multi-scale aggregation.

## Environment

The training and validation commands below were verified on Windows 11 with Python 3.11.15, PyTorch 2.6.0+cu124, CUDA 12.4, and an NVIDIA RTX 3050 Ti.

```bash
conda create -n menet python=3.11 pip -y
conda activate menet

# Install the CUDA 12.4 PyTorch build from the official PyTorch index.
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# The Tsinghua mirror is optional but faster in mainland China.
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -e . --no-deps
```

For a different CUDA version, install the matching PyTorch build first and then install the remaining dependencies.

## Pretrained checkpoint

Download `me-net.pt` from the [Google Drive folder](https://drive.google.com/drive/folders/1iiOtoAMWPsnER3bFPe82GhgzcpgVlxe7?usp=drive_link) and place it at:

```text
weights/me-net.pt
```

The checkpoint is hosted separately and is not tracked in this repository.

## Dataset

RGB images, DVS images, and YOLO labels must have identical relative names:

```text
dataset/
├── images/
│   ├── train/xxx.jpg
│   └── val/xxx.jpg
├── imagesDVS/
│   ├── train/xxx.jpg
│   └── val/xxx.jpg
├── labels/
│   ├── train/xxx.txt
│   └── val/xxx.txt
└── gareud.yaml
```

Each label file uses standard YOLO detection format:

```text
class_id x_center y_center width height
```

Example dataset YAML:

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val
nc: 4
names: ['DJI M350RTK', 'DJI Mavic 4 Pro', 'DJI Air3', 'DJI Avata2']
```

The loader finds each DVS image by replacing `images` with `imagesDVS` in the RGB path. Missing or differently named DVS files cause loading to fail.

## Training

Train from the ME-Net YAML configuration:

```bash
python train_dual.py --data /absolute/path/to/gareud.yaml --device 0
```

Fine-tune from the downloaded checkpoint:

```bash
python train_dual.py --data /absolute/path/to/gareud.yaml --model weights/me-net.pt --device 0
```

The default configuration uses 100 epochs, image size 640, and batch size 32. Override these values for smaller GPUs, for example:

```bash
python train_dual.py --data /absolute/path/to/gareud.yaml --epochs 3 --batch 8 --imgsz 320 --workers 2 --device 0
```

## Validation

```bash
python val_dual.py --data /absolute/path/to/gareud.yaml --weights weights/me-net.pt --batch 8 --imgsz 320 --workers 2 --device 0
```

The checkpoint contains four UAV classes and loads directly after it is placed in `weights/me-net.pt`:

```python
from ultralytics import YOLO

model = YOLO("weights/me-net.pt")
```

SHA256 of the checkpoint tested with this code:

```text
bb22810d634e46c6725db88a4934a924fdde94151cab7b1e35fb0b41d0c03762
```

## License

This repository is based on Ultralytics YOLO and is distributed under the AGPL-3.0 license. See `LICENSE`.
