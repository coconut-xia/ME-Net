# ME-Net

## Abstract

we propose **ME-Net**, a lightweight dual-stream detector tailored for efficient RGB-Event fusion. ME-Net combines RGB appearance cues and event motion cues through three components:

- **Motion-aware Event Refinement (MER)** suppresses background event noise and enhances motion-salient responses.
- **Asymmetric Cross-modal Fusion (ACF)** performs high-level feature interaction while preserving critical event motion cues when RGB quality degrades.
- **Adaptive Multi-scale Aggregation (AMA)** efficiently integrates features across scales for small-UAV detection.

Finally, we deploy the complete sensing-to-processing pipeline on an NVIDIA Jetson AGX Orin. The system achieves real-time performance of **52 FPS in G2A scenarios** and **34 FPS in A2A scenarios**.

## Citation

Citation information will be added when the paper becomes publicly available.

## License

License and third-party dependency details will be provided with the source-code release. ME-Net is implemented within the [Ultralytics](https://github.com/ultralytics/ultralytics) ecosystem; users should also follow the applicable upstream licensing terms.

## Contact

For questions about this project, please submit a GitHub issue.

## Thanks

Our implementation is mainly based on [Ultralytics](https://github.com/ultralytics/ultralytics). We thank the authors and contributors for their work.
