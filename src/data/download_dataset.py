"""
Download the Car Parts dataset from Roboflow.

Usage:
    python src/data/download_dataset.py

You will need a Roboflow API key. Get one at https://roboflow.com
Set it as an environment variable: export ROBOFLOW_API_KEY=your_key_here
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def download_dataset(api_key: str, output_dir: str = "data/splits"):
    """Download the car parts dataset from Roboflow."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("Please install roboflow: pip install roboflow")
        return

    rf = Roboflow(api_key=api_key)

    # Access the car parts dataset
    # Update workspace and project names based on the actual Roboflow URL
    project = rf.workspace("team-data").project("car-parts-ybiev")

    # Download the latest version in YOLOv8 format
    version = project.version(1)  # Update version number as needed
    dataset = version.download("yolov8", location=output_dir)

    print(f"Dataset downloaded to: {output_dir}")
    print(f"Number of classes: {dataset.n_classes if hasattr(dataset, 'n_classes') else 'Check dataset.yaml'}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Download Car Parts dataset")
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("ROBOFLOW_API_KEY"),
        help="Roboflow API key",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/splits",
        help="Output directory for dataset",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Error: No API key provided.")
        print("Set ROBOFLOW_API_KEY environment variable or use --api-key flag.")
        return

    download_dataset(args.api_key, args.output_dir)


if __name__ == "__main__":
    main()
