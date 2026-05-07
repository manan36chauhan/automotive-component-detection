"""
Evaluate trained YOLO models on the test set.

Usage:
    python src/evaluation/evaluate.py --weights runs/train/yolov8s_baseline/weights/best.pt
"""

import argparse
import json
from pathlib import Path
from ultralytics import YOLO


def evaluate_model(weights_path: str, data_config: str, imgsz: int = 640):
    """Evaluate a trained YOLO model and return metrics."""
    model = YOLO(weights_path)

    # Run validation on test set
    results = model.val(
        data=data_config,
        imgsz=imgsz,
        split="test",
        save_json=True,
        plots=True,
    )

    # Extract key metrics
    metrics = {
        "model": weights_path,
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "per_class_ap50": {
            name: float(ap)
            for name, ap in zip(results.names.values(), results.box.ap50)
        },
    }

    return metrics


def save_metrics(metrics: dict, output_path: str):
    """Save evaluation metrics to JSON file."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with open(output, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Metrics saved to: {output}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights (.pt)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/dataset.yaml",
        help="Path to dataset config",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for metrics JSON",
    )
    args = parser.parse_args()

    metrics = evaluate_model(args.weights, args.data, args.imgsz)

    # Print results
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"mAP@0.5:      {metrics['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['mAP50_95']:.4f}")
    print(f"Precision:     {metrics['precision']:.4f}")
    print(f"Recall:        {metrics['recall']:.4f}")
    print("\nPer-class AP@0.5:")
    for cls, ap in metrics["per_class_ap50"].items():
        print(f"  {cls}: {ap:.4f}")

    # Save metrics
    if args.output is None:
        model_name = Path(args.weights).parent.parent.name
        args.output = f"results/metrics/{model_name}_metrics.json"

    save_metrics(metrics, args.output)


if __name__ == "__main__":
    main()
