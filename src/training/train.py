"""
Training pipeline for YOLO-based automotive component detection.

Usage:
    python src/training/train.py --config configs/yolov8_config.yaml
    python src/training/train.py --model yolov8s.pt --epochs 100 --imgsz 640
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train(config: dict):
    """Train YOLO model with given configuration."""
    # Load model (pretrained or from scratch)
    model_name = config.get("model", "yolov8s.pt")
    model = YOLO(model_name)

    print(f"Model: {model_name}")
    print(f"Dataset: {config.get('data')}")
    print(f"Epochs: {config.get('epochs', 100)}")
    print(f"Image size: {config.get('imgsz', 640)}")
    print("-" * 50)

    # Train
    results = model.train(
        data=config.get("data", "configs/dataset.yaml"),
        epochs=config.get("epochs", 100),
        batch=config.get("batch", 16),
        imgsz=config.get("imgsz", 640),
        patience=config.get("patience", 20),
        device=config.get("device", 0),
        optimizer=config.get("optimizer", "AdamW"),
        lr0=config.get("lr0", 0.001),
        lrf=config.get("lrf", 0.01),
        momentum=config.get("momentum", 0.937),
        weight_decay=config.get("weight_decay", 0.0005),
        augment=config.get("augment", True),
        hsv_h=config.get("hsv_h", 0.015),
        hsv_s=config.get("hsv_s", 0.7),
        hsv_v=config.get("hsv_v", 0.4),
        degrees=config.get("degrees", 10.0),
        translate=config.get("translate", 0.1),
        scale=config.get("scale", 0.5),
        fliplr=config.get("fliplr", 0.5),
        mosaic=config.get("mosaic", 1.0),
        project=config.get("project", "runs/train"),
        name=config.get("name", "experiment"),
        save=config.get("save", True),
        save_period=config.get("save_period", 10),
    )

    print("\nTraining complete!")
    print(f"Results saved to: {results.save_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/yolov8_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument("--model", type=str, help="Override model name")
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--imgsz", type=int, help="Override image size")
    parser.add_argument("--batch", type=int, help="Override batch size")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Apply command-line overrides
    if args.model:
        config["model"] = args.model
    if args.epochs:
        config["epochs"] = args.epochs
    if args.imgsz:
        config["imgsz"] = args.imgsz
    if args.batch:
        config["batch"] = args.batch

    train(config)


if __name__ == "__main__":
    main()
