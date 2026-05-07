# Improving Detection of Visually Challenging Automotive Components: An Empirical Evaluation of YOLO Training Strategies

## Overview
This project investigates why certain automotive component categories are significantly harder to detect than others in multi-class object detection, and evaluates which standard training strategies most effectively improve detection of these challenging classes.

A baseline experiment on a 50-class car parts dataset reveals a **32 percentage point performance gap** between the best-performing class (Clutch Plate, 0.994 mAP) and the worst (Ignition Coil, 0.675 mAP) — despite Ignition Coil having the most training samples. This indicates that **visual complexity, not data scarcity**, is the primary barrier to reliable detection.

**Dissertation Module:** M598 – Master Dissertation (60 ECTS)  
**University:** Gisma University of Applied Sciences  
**Department:** Computer and Data Sciences (CDS)  
**Supervisor:** Professor Mohammad

## Research Questions
1. **RQ1:** Which automotive component categories are most challenging to detect, and what visual characteristics contribute to their difficulty?
2. **RQ2:** How do model architecture size, input resolution, and transfer learning affect detection accuracy for visually challenging components?
3. **RQ3:** Which data augmentation strategies most effectively improve detection of hard-to-detect automotive parts?

## Key Finding
Five classes consistently fall below 0.85 mAP@0.5:

| Class | mAP@0.5 | Recall | Training Samples | Key Issue |
|-------|---------|--------|-----------------|-----------|
| Ignition Coil | 0.675 | 0.491 | 162 (highest) | Most data, worst performance |
| Gas Cap | 0.717 | 0.441 | 116 | Lowest recall in dataset |
| Distributor | 0.792 | 0.553 | 108 | High precision, low recall |
| Overflow Tank | 0.810 | 0.646 | 152 | Both metrics weak |
| Oil Pressure Sensor | 0.848 | 0.686 | 102 | Model is conservative |

## Dataset
- **Source:** [Car Parts Dataset – Roboflow](https://universe.roboflow.com/team-data/car-parts-ybiev)
- **Images:** 8,739 total (6,118 train / 3,146 val / 2,534 test)
- **Classes:** 50 automotive component categories
- **Annotations:** YOLO format bounding boxes, 1 annotation per image
- **Class distribution:** 86 (Piston) to 162 (Ignition Coil) samples per class (1.9x imbalance ratio)

## Experiments

| # | Experiment | Variable Tested | RQ | Status |
|---|-----------|----------------|-----|--------|
| 1 | Baseline YOLOv8s | Establish per-class performance | RQ1 | Done |
| 2 | Architecture comparison | YOLOv8n / v8s / v8m | RQ2 | Done |
| 3 | Resolution impact | 320 / 640 / 800 px | RQ2 | Planned |
| 4 | Augmentation strategies | None / standard / advanced (mosaic+mixup) | RQ3 | Planned |
| 5 | Transfer learning | COCO pretrained vs from scratch | RQ2 | Planned |
| 6 | YOLO26 comparison | YOLOv8s vs YOLO26s (cross-generational) | RQ2 | Planned |

## Baseline Results (Experiment 1)

**Model:** YOLOv8s | **Image size:** 640px | **Pretrained:** COCO | **Epochs:** 100

| Metric | Score |
|--------|-------|
| mAP@0.5 | 0.936 |
| mAP@0.5:0.95 | 0.935 |
| Precision | 0.924 |
| Recall | 0.841 |

## Methods
- **Framework:** PyTorch + Ultralytics
- **Detection Architectures:** YOLOv8 (n/s/m variants), YOLO26
- **Transfer Learning:** Models initialised with COCO pretrained weights
- **Augmentation:** Geometric transforms, HSV colour jittering, mosaic, mixup
- **Evaluation:** Precision, Recall, mAP@0.5, mAP@0.5:0.95 — measured both overall AND per-class
- **Training Platform:** Google Colab (T4 GPU) and Kaggle

## Project Structure
```
automotive-component-detection/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
├── configs/
│   ├── yolov8_config.yaml
│   └── dataset.yaml
├── src/
│   ├── data/
│   │   └── download_dataset.py
│   ├── training/
│   │   └── train.py
│   └── evaluation/
│       ├── evaluate.py
│       ├── compare_models.py
│       └── visualize_results.py
├── notebooks/
│   ├── 01_colab_training.ipynb
│   └── 02_remaining_experiments.ipynb
├── results/
│   ├── metrics/
│   └── visualizations/
└── docs/
    └── experiment_log.md
```

## Setup and Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended) or Google Colab / Kaggle
- Git

### Installation
```bash
git clone https://github.com/manan36chauhan/automotive-component-detection.git
cd automotive-component-detection
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Download Dataset
```bash
cp .env.example .env
# Edit .env and add your Roboflow API key
python src/data/download_dataset.py
```

### Run Training (Colab/Kaggle)
Upload the notebooks from `notebooks/` to Google Colab or Kaggle. Enable T4 GPU runtime and run cells in order.

## Reproducibility
- **Python:** 3.12
- **PyTorch:** 2.10.0+cu128
- **Ultralytics:** 8.4.30
- **CUDA:** T4 GPU (14.9 GB VRAM)
- **Random seed:** 0 (deterministic=True)
- **Dataset:** Publicly available on Roboflow

## License
This project is developed as part of an academic dissertation at Gisma University of Applied Sciences. All rights reserved.

## Acknowledgements
- Gisma University of Applied Sciences
- Dataset: [Roboflow - Car Parts (team-data/car-parts-ybiev)](https://universe.roboflow.com/team-data/car-parts-ybiev)
