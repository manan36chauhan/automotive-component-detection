"""
Visualize detection results with bounding boxes on input images.

Usage:
    python src/evaluation/visualize_results.py \
        --weights runs/train/yolov8s_baseline/weights/best.pt \
        --source data/splits/test/images \
        --output results/visualizations/detections
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def visualize_detections(
    weights_path: str,
    source: str,
    output_dir: str = "results/visualizations/detections",
    conf_threshold: float = 0.25,
    imgsz: int = 640,
):
    """Run inference and save images with bounding boxes."""
    model = YOLO(weights_path)

    results = model.predict(
        source=source,
        conf=conf_threshold,
        imgsz=imgsz,
        save=True,
        save_txt=True,
        project=output_dir,
        name="predictions",
        line_width=2,
        show_labels=True,
        show_conf=True,
    )

    print(f"\nDetection results saved to: {output_dir}/predictions")
    print(f"Total images processed: {len(results)}")

    # Print summary
    for r in results[:5]:  # Show first 5 results
        boxes = r.boxes
        print(f"\n{Path(r.path).name}: {len(boxes)} detections")
        for box in boxes:
            cls = int(box.cls)
            conf = float(box.conf)
            name = r.names[cls]
            print(f"  - {name}: {conf:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Visualize detections")
    parser.add_argument("--weights", type=str, required=True, help="Model weights")
    parser.add_argument("--source", type=str, required=True, help="Image source path")
    parser.add_argument("--output", type=str, default="results/visualizations/detections")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    visualize_detections(args.weights, args.source, args.output, args.conf, args.imgsz)


if __name__ == "__main__":
    main()
