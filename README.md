# ME-Net

Official project repository for **ME-Net**, the lightweight RGB-Event UAV detector introduced in:

> **Full-Stack UAV Detection Pipeline from Precise RGB-Event Sensing to Efficient Edge Processing**

> [!IMPORTANT]
> This repository is under preparation. The cleaned network implementation, pretrained weights, training and evaluation instructions, and dataset access information will be released after internal organization.

## Overview

ME-Net is a lightweight dual-stream detector designed to combine RGB appearance cues with event-camera motion cues under edge-computing constraints. It is part of a full-stack UAV detection pipeline covering precise RGB-Event sensing, the GAREUD benchmark, multimodal detection, and deployment on an NVIDIA Jetson AGX Orin.

The system targets difficult ground-to-air (G2A) and air-to-air (A2A) conditions, including small targets, fast motion, low or excessive illumination, background clutter, platform vibration, and ego-motion.

## Method

ME-Net contains three main components:

- **Motion-aware Event Refinement (MER)** suppresses low-energy background activity and enhances motion-salient event responses.
- **Asymmetric Cross-modal Fusion (ACF)** performs high-level RGB-Event interaction while preserving critical event-motion information when RGB quality degrades.
- **Adaptive Multi-scale Aggregation (AMA)** uses efficient bidirectional feature aggregation to combine multi-scale features for small-UAV detection.

```text
RGB frames ───────────────► RGB backbone ────────┐
                                                 ├─► ACF ─► AMA ─► Detection heads
Event representation ─► MER ─► Event backbone ──┘
```

The detector is implemented within the Ultralytics framework and produces predictions at multiple feature scales.

## Full-Stack Pipeline

The accompanying work integrates:

1. **Precise sensing** — co-axial RGB-Event optics and microsecond-level hardware triggering.
2. **GAREUD** — a Ground-Aerial RGB-Event UAV Detection dataset with real-world and synthetic subsets, G2A and A2A viewpoints, and diverse illumination and motion conditions.
3. **Efficient fusion** — ME-Net combines complementary appearance and motion information through MER, ACF, and AMA.
4. **Edge deployment** — TensorRT FP16 inference on NVIDIA Jetson AGX Orin.

The complete pipeline reaches **52 FPS in G2A scenarios** and **34 FPS in A2A scenarios** on the Jetson AGX Orin platform.

## Release Status

| Item | Status |
| --- | --- |
| Project README | Available |
| ME-Net source code | In preparation |
| Training and evaluation scripts | In preparation |
| Pretrained checkpoints | In preparation |
| GAREUD dataset and access instructions | In preparation |
| Paper and citation metadata | In preparation |

## Citation

Citation information will be added when the paper becomes publicly available.

## License

License and third-party dependency details will be provided together with the source-code release. ME-Net is implemented within the Ultralytics ecosystem; users should also follow the applicable upstream licensing terms.

## Contact

For questions about this project, please open a GitHub issue after the code release.
