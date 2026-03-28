# Experiment Log

## Overview
This document tracks all experiments conducted for the dissertation:
**"Deep Learning-Based Detection and Localization of Automotive Components"**

---

## Experiment 1: Baseline YOLOv8s
- **Date:** YYYY-MM-DD
- **Model:** YOLOv8s (pretrained on COCO)
- **Image Size:** 640x640
- **Epochs:** 100
- **Batch Size:** 16
- **Augmentation:** Default YOLO augmentations
- **Results:**
  - mAP@0.5: -
  - mAP@0.5:0.95: -
  - Precision: -
  - Recall: -
- **Notes:** Baseline experiment to establish reference performance.

---

## Experiment 2: Architecture Comparison (RQ2)
- **Date:** YYYY-MM-DD
- **Models:** YOLOv8n, YOLOv8s, YOLOv8m
- **Image Size:** 640x640
- **Epochs:** 100
- **Results:**

| Model    | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Params (M) |
|----------|---------|---------------|-----------|--------|------------|
| YOLOv8n  | -       | -             | -         | -      | -          |
| YOLOv8s  | -       | -             | -         | -      | -          |
| YOLOv8m  | -       | -             | -         | -      | -          |

- **Notes:**

---

## Experiment 3: Image Resolution Impact (RQ2)
- **Date:** YYYY-MM-DD
- **Model:** YOLOv8s
- **Image Sizes:** 320, 480, 640, 800
- **Results:**

| Resolution | mAP@0.5 | mAP@0.5:0.95 | Training Time |
|------------|---------|---------------|---------------|
| 320x320    | -       | -             | -             |
| 480x480    | -       | -             | -             |
| 640x640    | -       | -             | -             |
| 800x800    | -       | -             | -             |

- **Notes:**

---

## Experiment 4: Augmentation Strategies (RQ3)
- **Date:** YYYY-MM-DD
- **Model:** YOLOv8s
- **Configurations:**
  - No augmentation
  - Standard augmentation (rotation, flip, brightness)
  - Advanced augmentation (mosaic, mixup, copy-paste)
- **Results:**

| Augmentation | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|-------------|---------|---------------|-----------|--------|
| None        | -       | -             | -         | -      |
| Standard    | -       | -             | -         | -      |
| Advanced    | -       | -             | -         | -      |

- **Notes:**

---

## Experiment 5: Transfer Learning vs From Scratch (RQ2)
- **Date:** YYYY-MM-DD
- **Model:** YOLOv8s
- **Configurations:** Pretrained (COCO) vs Random Initialization
- **Results:**

| Init         | mAP@0.5 | mAP@0.5:0.95 | Convergence (epochs) |
|-------------|---------|---------------|----------------------|
| Pretrained  | -       | -             | -                    |
| From Scratch| -       | -             | -                    |

- **Notes:**
