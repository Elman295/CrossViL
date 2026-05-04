# CrossViL: Remote Sensing Change Detection with Cross mLSTM

<div align="center">

[![ICIP 2026](https://img.shields.io/badge/ICIP-2026-blue.svg)](https://2026.ieeeicip.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**[Elman Ghazaei](https://github.com/), [Erchan Aptoula](https://github.com/)**

Faculty of Engineering and Natural Sciences (VPALab), Sabanci University, Istanbul, Türkiye

*Accepted at IEEE ICIP 2026*

</div>

---

## Overview



<div align="center">
  <img src="https://github.com/user-attachments/assets/4a08ea7b-0fae-4cb0-bd5d-12a44991c8f8" width="90%" alt="CrossViL Architecture"/>
  <p><em>Overall framework of CrossViL. Left: Full pipeline. Middle: CrossViL block with Cross-mLSTM units. Right: Detailed Cross-mLSTM structure.</em></p>
</div>

**CrossViL** is a novel architecture for remote sensing change detection (CD) that introduces a **cross-interaction fusion strategy** using Vision mLSTM (xLSTM). Query representations are exchanged between pre- and post-event features, enabling cross-guided temporal reasoning without the CUDA constraints of Mamba-based methods.

### Key Contributions

- 📐 **Cross-mLSTM block**: Bidirectional query exchange between pre/post-event feature streams for explicit cross-temporal interaction
- 🔁 **CrossViL encoder**: Multi-stage fusion pipeline integrating Cross-mLSTM blocks with flip-augmented feature processing
- 🏆 **State-of-the-art results**: Outperforms CNN-, ViT-, Mamba-, and xLSTM-based methods on LEVIR-CD and WHU-CD benchmarks
- ⚡ **Deployment-friendly**: No strict CUDA configuration requirements (unlike Mamba-based models)

---

## Results

### LEVIR-CD

| Model | Venue | F1 (%) | OA (%) | IoU (%) |
|---|---|---|---|---|
| BIT | TGRS'22 | 89.31 | 98.92 | 80.68 |
| ChangeFormer | IGARSS'22 | 90.40 | 99.04 | 82.48 |
| DMINet | TGRS'23 | 90.71 | 99.07 | 82.99 |
| ICIF-Net | TGRS'22 | 91.18 | 99.12 | 83.85 |
| ChangeMamba | TGRS'24 | 90.51 | 99.04 | 82.67 |
| CDMamba | arxiv'25 | 90.75 | 99.06 | 83.07 |
| STDF-CD | JSTARS'25 | 91.92 | 99.12 | 85.05 |
| CDxLSTM | GRSL'25 | 90.89 | 99.07 | 83.30 |
| **CrossViL (Ours)** | **ICIP'26** | **93.65** | **98.80** | **88.60** |

### WHU-CD

| Model | Venue | F1 (%) | OA (%) | IoU (%) |
|---|---|---|---|---|
| BIT | TGRS'22 | 87.47 | 98.81 | 77.73 |
| ChangeFormer | IGARSS'22 | 86.88 | 98.76 | 76.81 |
| DMINet | TGRS'23 | 91.49 | 99.19 | 84.31 |
| ChangeCLIP | ISPRS'24 | 94.82 | 99.52 | 90.15 |
| Change3D | CVPR'25 | 94.56 | 99.57 | 89.69 |
| STDF-CD | JSTARS'25 | 94.90 | 99.47 | 90.29 |
| CDxLSTM | GRSL'25 | 92.58 | 99.33 | 86.19 |
| **CrossViL (Ours)** | **ICIP'26** | **95.71** | **99.37** | **92.04** |

---

## Method

### Architecture Overview

CrossViL receives pre-event (*I*<sub>pre</sub>) and post-event (*I*<sub>post</sub>) image pairs and processes them through three components:

1. **Vision Backbone** — Shared pretrained ResNet-18 extracts feature representations *x*<sub>pre</sub> and *x*<sub>post</sub>
2. **CrossViL Encoder** — A stack of Cross-mLSTM blocks that iteratively refine and fuse bi-temporal features
3. **Decoder** — Progressive upsampling with skip connections to produce the final binary change map

### Cross-mLSTM

The core of CrossViL is the **Cross-mLSTM** module. Given pre- and post-event features *z*<sub>pre</sub> and *z*<sub>post</sub>, separate Q/K/V projections are computed for each stream. The key operation is a **query swap**:

$$\tilde{Q}_{pre} = Q_{post}, \quad \tilde{Q}_{post} = Q_{pre}$$

Each stream then runs an mLSTM cell with the swapped query, making the two pathways mutually conditioned. The final fused representation is:

$$H_t = h_t^{pre} \oplus h_t^{post}$$

This design forces the network to reason about temporal differences explicitly, rather than processing each time point in isolation.

### Loss Function

$$\mathcal{L}_{seg} = \mathcal{L}_{CE} + \mathcal{L}_{lov}$$

Cross-entropy loss combined with Lovász-Softmax loss to handle class imbalance between changed and unchanged pixels.

---

## Ablation Studies

### Number of CrossViL Blocks (WHU-CD)

| # Blocks | OA (%) | F1 (%) | IoU (%) |
|---|---|---|---|
| 0 (Baseline) | 98.42 | 90.58 | 83.30 |
| 1 | 99.30 | 95.29 | 91.34 |
| 2 | 99.33 | 95.45 | 91.59 |
| **3** | **99.37** | **95.71** | **92.04** |
| 4 | 99.34 | 95.33 | 91.73 |

### Hidden Dimension (WHU-CD)

| Dims | F1 (%) | OA (%) | IoU (%) | Params (M) |
|---|---|---|---|---|
| 32 | 91.36 | 98.71 | 85.09 | 14.50 |
| 64 | 93.37 | 99.06 | 88.18 | 14.81 |
| 128 | 95.32 | 99.30 | 91.38 | 15.88 |
| **256** | **95.71** | **99.37** | **92.04** | 24.17 |
| 512 | 95.36 | 99.32 | 91.45 | 34.67 |

---

## Getting Started

> ⚠️ **Code will be released publicly upon acceptance.** This repository will be updated with full training and evaluation code.

### Requirements

```bash
# Coming soon
pip install -r requirements.txt
```

### Datasets

**LEVIR-CD**
- 637 image pairs at 1024×1024 resolution
- Cropped to 256×256 patches for training/evaluation
- [Download](https://justchenhao.github.io/LEVIR/)

**WHU-CD**
- High-resolution (0.3 m/pixel) aerial imagery, ~20.5 km²
- [Download](http://gpcv.whu.edu.cn/data/building_dataset.html)

### Training

```bash
# Coming soon
python train.py --dataset LEVIR-CD --epochs 100 --lr 0.0001 --dims 256 --num_blocks 3
```

### Evaluation

```bash
# Coming soon
python evaluate.py --dataset WHU-CD --checkpoint checkpoints/crossvil_best.pth
```

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{ghazaei2026crossvil,
  title     = {Remote Sensing Change Detection with Cross mLSTM},
  author    = {Ghazaei, Elman and Aptoula, Erchan},
  booktitle = {IEEE International Conference on Image Processing (ICIP)},
  year      = {2026}
}
```

---

## Related Work

- [CDxLSTM](https://ieeexplore.ieee.org/document/) — xLSTM for CD (GRSL'25)
- [STDF-CD](https://ieeexplore.ieee.org/document/) — STxLSTM for CD (JSTARS'25)
- [ChangeMamba](https://ieeexplore.ieee.org/document/) — Mamba-based CD (TGRS'24)
- [VisionLSTM](https://arxiv.org/abs/2406.04303) — xLSTM as vision backbone
- [xLSTM](https://arxiv.org/abs/2405.04517) — Extended LSTM (NeurIPS'24)

---

## Acknowledgments

This study was supported by the Scientific and Technological Research Council of Türkiye (TÜBİTAK) under Grant No. **123R108**.

---

<div align="center">
  <sub>VPALab · Sabanci University · Istanbul, Türkiye</sub>
</div>
