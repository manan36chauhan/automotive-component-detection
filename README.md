# Deep Learning-Based Detection and Localization of Automotive Components

## Overview
This project investigates the capability of deep learning-based object detection models to accurately detect and localize automotive components in images. It focuses on evaluating the performance of modern YOLO-based detection architectures and identifying training approaches that improve detection accuracy.

**Dissertation Module:** M598 – Master Dissertation (60 ECTS)  
**University:** Gisma University of Applied Sciences  
**Department:** Computer and Data Sciences (CDS)

## Research Questions
1. How effectively can deep learning-based object detection models detect and localize car parts in images?
2. How does detection performance vary across different model architectures and training configurations?
3. Which data preparation and augmentation techniques can improve detection accuracy for car parts?

## Dataset
- **Source:** [Car Parts Dataset – Roboflow](https://universe.roboflow.com/team-data/car-parts-ybiev)
- **Format:** Annotated images with bounding boxes
- **Split:** Training / Validation / Testing

## Methods
- **Framework:** PyTorch
- **Detection Architecture:** YOLOv8 / YOLOv9 / YOLOv11 (comparative study)
- **Transfer Learning:** Models initialized with pretrained weights (COCO)
- **Augmentation:** Rotation, brightness adjustment, scaling, flipping, mosaic
- **Evaluation Metrics:** Precision, Recall, mAP@0.5, mAP@0.5:0.95

## Project Structure
```
automotive-component-detection/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── configs/
│   ├── yolov8_config.yaml       # YOLOv8 training configuration
│   ├── yolov9_config.yaml       # YOLOv9 training configuration
│   └── dataset.yaml             # Dataset paths and class definitions
├── data/
│   ├── raw/                     # Original dataset (not tracked by git)
│   ├── processed/               # Preprocessed images
│   └── splits/                  # Train/val/test splits
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── download_dataset.py  # Script to download from Roboflow
│   │   ├── preprocess.py        # Image preprocessing pipeline
│   │   └── augmentation.py      # Custom augmentation strategies
│   ├── models/
│   │   ├── __init__.py
│   │   └── detector.py          # Model loading and configuration
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train.py             # Training pipeline
│   │   └── hyperparameters.py   # Hyperparameter configurations
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate.py          # Evaluation metrics computation
│   │   ├── compare_models.py    # Cross-model comparison
│   │   └── visualize_results.py # Bounding box visualization
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Experiment logging
│       └── helpers.py           # Utility functions
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_experiments.ipynb
│   └── 03_results_analysis.ipynb
├── results/
│   ├── metrics/                 # Saved evaluation metrics
│   └── visualizations/          # Detection output images
├── docs/
│   └── experiment_log.md        # Record of all experiments
└── tests/
    └── test_pipeline.py         # Unit tests
```

## Setup and Installation

### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- Git

### Installation
```bash
# Clone the repository
git clone https://github.com/<your-username>/automotive-component-detection.git
cd automotive-component-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download Dataset
```bash
python src/data/download_dataset.py
```

## Running Experiments

### Train a model
```bash
python src/training/train.py --config configs/yolov8_config.yaml
```

### Evaluate a model
```bash
python src/evaluation/evaluate.py --weights runs/best.pt --data configs/dataset.yaml
```

### Compare models
```bash
python src/evaluation/compare_models.py
```

## Results
Results and comparisons will be documented in `results/` and in the experiment log at `docs/experiment_log.md`.

| Model   | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Inference (ms) |
|---------|---------|---------------|-----------|--------|----------------|
| YOLOv8n | -       | -             | -         | -      | -              |
| YOLOv8s | -       | -             | -         | -      | -              |
| YOLOv9  | -       | -             | -         | -      | -              |

## License
This project is developed as part of an academic dissertation. All rights reserved.

## Acknowledgments
- Gisma University of Applied Sciences
- Supervisor: 
- Dataset: [Roboflow - Car Parts](https://universe.roboflow.com/team-data/car-parts-ybiev)
