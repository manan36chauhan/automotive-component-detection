# Improving Detection of Visually Challenging Automotive Components: An Empirical Evaluation of YOLO Training Strategies

## Overview
This repository contains the complete experimental pipeline for a Master's dissertation investigating why certain automotive component categories consistently underperform in object detection despite having sufficient training data, and evaluating which standard YOLO training strategies most effectively close these per-class performance gaps.

A baseline YOLOv8s model achieves **0.9417 mAP@0.5** overall on a 50-class automotive dataset, but five classes fall below the 0.85 reliability threshold. The central finding is that **Ignition Coil has the most training samples (162) yet the worst accuracy (0.792 mAP)** — statistical analysis (Pearson r = -0.160, p = 0.27) confirms that **visual complexity, not data scarcity**, drives per-class detection failures.

**Dissertation Module:** M598 - Master Dissertation (60 ECTS)  
**University:** Gisma University of Applied Sciences  
**Department:** Computer and Data Sciences (CDS)  
**Supervisor:** Prof. Dr. Mohammad Mahdavi  
**Author:** Manan Chauhan  
**Date:** June 2026

## Research Questions
1. **RQ1:** Which automotive component categories are most challenging to detect, and what visual characteristics contribute to their difficulty?
2. **RQ2:** How do model architecture, input resolution, and transfer learning affect detection accuracy for visually challenging components?
3. **RQ3:** Which data augmentation strategies most effectively improve detection of hard-to-detect automotive parts?

## Hard Classes Identified

| Class | mAP@0.5 | Precision | Recall | Samples | Visual Challenge |
|-------|---------|-----------|--------|---------|-----------------|
| Ignition Coil | 0.792 | 0.755 | 0.696 | 162 | High intra-class variation |
| Gas Cap | 0.696 | 0.802 | 0.483 | 116 | Minimal distinguishing features |
| Distributor | 0.769 | 0.738 | 0.711 | 108 | Visually similar to Ignition Coil |
| Overflow Tank | 0.834 | 0.808 | 0.688 | 152 | Reflective metallic surfaces |
| Oil Pressure Sensor | 0.932 | 0.880 | 0.861 | 102 | Small size relative to frame |

## Dataset
- **Source:** [Car Parts Dataset - Roboflow](https://universe.roboflow.com/team-data/car-parts-ybiev)
- **Images:** 8,739 total (6,118 train / 3,146 val / 2,534 test)
- **Classes:** 50 automotive component categories
- **Annotations:** YOLO format bounding boxes, 1 annotation per image
- **Class range:** 86 (Piston) to 162 (Ignition Coil) samples per class (1.9x imbalance ratio)

## Experiments - All Complete

| # | Experiment | Variable | Configurations | RQ |
|---|-----------|----------|----------------|-----|
| 1 | Baseline | Reference point | YOLOv8s, 640px, COCO pretrained | RQ1 |
| 2 | Architecture | Model capacity | YOLOv8n / v8s / v8m | RQ2 |
| 3 | Resolution | Input size | 320 / 640 / 800 px | RQ2 |
| 4 | Augmentation | Training diversity | None / Standard / Advanced (mosaic+mixup) | RQ3 |
| 5 | Transfer learning | Weight init | COCO pretrained / From scratch | RQ2 |
| 6 | Cross-generational | Architecture | YOLOv8s vs YOLO26s | RQ2 |

Total compute time: approximately 25 hours.

## Results

### Overall Performance

| Experiment | Configuration | mAP@0.5 | Time (min) |
|-----------|--------------|---------|------------|
| Exp 1 Baseline | YOLOv8s, 640px, COCO | 0.9417 | 102.5 |
| Exp 2 v8n | YOLOv8n | 0.9155 | 67.3 |
| Exp 2 v8m | YOLOv8m | 0.9555 | 202.9 |
| **Exp 3 320px** | **YOLOv8s, 320px** | **0.9663** | **42.5** |
| Exp 3 800px | YOLOv8s, 800px | 0.9157 | 183.0 |
| Exp 4 No aug | All augmentation off | 0.8443 | 106.2 |
| Exp 4 Standard | Geometric + colour | 0.9122 | 104.3 |
| Exp 4 Advanced | + mosaic + mixup | 0.9437 | 108.0 |
| Exp 5 Scratch | Random init, 80 epochs | 0.8423 | 169.5 |
| Exp 6 YOLO26s | YOLO26s with STAL | 0.9483 | 131.7 |

### Best Configuration Per Hard Class

| Hard Class | Best Experiment | Best mAP@0.5 | Baseline | Gain |
|------------|----------------|-------------|----------|------|
| Ignition Coil | 320px (Exp 3) | 0.854 | 0.792 | +0.062 |
| Gas Cap | 320px (Exp 3) | 0.809 | 0.696 | +0.113 |
| Distributor | 320px (Exp 3) | 0.888 | 0.769 | +0.119 |
| Overflow Tank | 320px (Exp 3) | 0.904 | 0.834 | +0.070 |
| Oil Pressure Sensor | YOLO26s (Exp 6) | 0.970 | 0.932 | +0.038 |

### Key Findings

1. **More data does not equal better detection.** Pearson r = -0.160, p = 0.27. No correlation between training sample count and per-class mAP.

2. **320px outperforms 640px and 800px** for all hard classes. Dataset images are low-resolution; upscaling adds artefacts. Training is also 4.3x faster.

3. **Advanced augmentation (mosaic + mixup) is the most impactful strategy.** +10 percentage points overall. Distributor gains 32.2 points.

4. **COCO pretraining is essential.** Without it, hard classes lose 22-32 points. Ignition Coil drops from 0.792 to 0.468.

5. **YOLO26 achieves best overall mAP (0.9483)** but does not universally solve per-class failures. Gas Cap +7.4%, but Ignition Coil -5.4%.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | SGD |
| Learning rate | 0.01 |
| Batch size | 16 (8 for 800px) |
| Image size | 640px (default) |
| Epochs | 50 (80 for from-scratch) |
| Early stopping | 10 (15 for from-scratch) |
| Random seed | 42 |
| Deterministic | True |
| Pretrained | COCO (unless stated) |

## Setup and Reproduction

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU or Kaggle/Colab (T4 GPU)

### Installation
```bash
git clone https://github.com/manan36chauhan/automotive-component-detection.git
cd automotive-component-detection
pip install ultralytics roboflow matplotlib scipy
```

### Run Experiments
Upload the experiment notebook to Kaggle. Enable T4 GPU and run cells sequentially. Each experiment saves results as JSON.

### Generate Figures
Upload `thesis_figures_generator.ipynb` to Kaggle. No GPU needed. Produces all dissertation figures in 30 seconds.

## Reproducibility
- **Python:** 3.12
- **PyTorch:** 2.x with CUDA
- **Ultralytics:** 8.4.30
- **Hardware:** Kaggle T4 GPU (2x 15GB VRAM, 30 hrs/week)
- **Random seed:** 42 (deterministic=True)
- **Dataset:** Publicly available on Roboflow

## Citation
```
Chauhan, M. (2026) 'Improving Detection of Visually Challenging Automotive Components:
An Empirical Evaluation of YOLO Training Strategies', Master Dissertation,
Gisma University of Applied Sciences.
```

## Acknowledgements
- Gisma University of Applied Sciences, Department of Computer and Data Sciences
- Supervisor: Prof. Dr. Mohammad Mahdavi
- Dataset: [Roboflow - Car Parts (team-data/car-parts-ybiev)](https://universe.roboflow.com/team-data/car-parts-ybiev)
