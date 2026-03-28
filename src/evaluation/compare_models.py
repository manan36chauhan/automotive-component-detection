"""
Compare multiple trained YOLO models side by side.

Usage:
    python src/evaluation/compare_models.py \
        --weights runs/train/yolov8n/weights/best.pt \
                  runs/train/yolov8s/weights/best.pt \
                  runs/train/yolov8m/weights/best.pt
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO


def compare_models(weight_paths: list, data_config: str, imgsz: int = 640):
    """Evaluate and compare multiple models."""
    all_metrics = []

    for weights in weight_paths:
        print(f"\nEvaluating: {weights}")
        model = YOLO(weights)

        results = model.val(data=data_config, imgsz=imgsz, split="test")

        model_name = Path(weights).parent.parent.name
        metrics = {
            "Model": model_name,
            "mAP@0.5": round(float(results.box.map50), 4),
            "mAP@0.5:0.95": round(float(results.box.map), 4),
            "Precision": round(float(results.box.mp), 4),
            "Recall": round(float(results.box.mr), 4),
        }
        all_metrics.append(metrics)

    return pd.DataFrame(all_metrics)


def plot_comparison(df: pd.DataFrame, output_dir: str = "results/visualizations"):
    """Generate comparison charts."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    metrics_cols = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"]

    # Bar chart comparison
    fig, axes = plt.subplots(1, len(metrics_cols), figsize=(16, 5))
    for i, metric in enumerate(metrics_cols):
        axes[i].bar(df["Model"], df[metric], color=plt.cm.Set2(range(len(df))))
        axes[i].set_title(metric)
        axes[i].set_ylim(0, 1)
        axes[i].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison chart saved to: {output_dir}/model_comparison.png")


def main():
    parser = argparse.ArgumentParser(description="Compare YOLO models")
    parser.add_argument(
        "--weights",
        type=str,
        nargs="+",
        required=True,
        help="Paths to model weights",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/dataset.yaml",
        help="Dataset config path",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    df = compare_models(args.weights, args.data, args.imgsz)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(df.to_string(index=False))

    # Save results
    Path("results/metrics").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/metrics/model_comparison.csv", index=False)

    plot_comparison(df)


if __name__ == "__main__":
    main()
