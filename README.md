# Full-Stack UAV Detection Pipeline from Precise RGB-Event Sensing to Efficient Edge Processing

Haoji Xia, Siying Liu, Jiaqi Jin, Sihan Wang, Tong Huang, Xujie Han, Zikai Wang, Yuxin Zhong, Hanle Zheng, Chen Cheng, Hao Guo, and Lei Deng

Taiyuan University of Technology · Tsinghua University

---

🚀 🚀 🚀 **News**

- **June 22, 2026**: Project repository created and README released.
- The cleaned ME-Net implementation, pretrained checkpoints, training and evaluation instructions, and GAREUD access information are being prepared.

## Abstract

RGB-Event fusion offers strong potential for UAV detection, but real-world deployment is limited by the lack of precisely aligned multimodal datasets and efficient edge-processing methods. We present a full-stack RGB-Event UAV detection pipeline spanning precise sensing, dataset construction, multimodal modeling, and edge deployment.

First, we construct **GAREUD**, a Ground-Aerial RGB-Event UAV Detection dataset with real-world and synthetic subsets. Microsecond-level hardware triggering and a co-axial optical configuration provide precise temporal and spatial alignment. GAREUD covers ground-to-air (G2A) and air-to-air (A2A) viewpoints under diverse illumination and motion conditions.

Second, we propose **ME-Net**, a lightweight dual-stream detector tailored for efficient RGB-Event fusion. ME-Net combines RGB appearance cues and event motion cues through three components:

- **Motion-aware Event Refinement (MER)** suppresses background event noise and enhances motion-salient responses.
- **Asymmetric Cross-modal Fusion (ACF)** performs high-level feature interaction while preserving critical event motion cues when RGB quality degrades.
- **Adaptive Multi-scale Aggregation (AMA)** efficiently integrates features across scales for small-UAV detection.

Finally, we deploy the complete sensing-to-processing pipeline on an NVIDIA Jetson AGX Orin using TensorRT FP16. The system achieves real-time performance of **52 FPS in G2A scenarios** and **34 FPS in A2A scenarios**.

## Release Plan

- [x] Project repository and README
- [ ] ME-Net source code
- [ ] Training and evaluation scripts
- [ ] Pretrained checkpoints
- [ ] GAREUD dataset and access instructions
- [ ] Paper and citation metadata

## Citation

Citation information will be added when the paper becomes publicly available.

## License

License and third-party dependency details will be provided with the source-code release. ME-Net is implemented within the [Ultralytics](https://github.com/ultralytics/ultralytics) ecosystem; users should also follow the applicable upstream licensing terms.

## Contact

For questions about this project, please submit a GitHub issue.

## Thanks

Our implementation is mainly based on [Ultralytics](https://github.com/ultralytics/ultralytics). We thank the authors and contributors for their work.
