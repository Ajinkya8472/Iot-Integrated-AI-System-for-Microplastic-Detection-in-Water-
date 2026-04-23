"""
Microplastic Detection - Inference Script
==========================================
Run YOLOv5/YOLO11 inference on images or video for microplastic detection.

Usage:
    python src/detect.py --source path/to/image.jpg --weights results/weights/best.pt
    python src/detect.py --source path/to/video.mp4 --weights results/weights/best.pt
    python src/detect.py --source 0 --weights results/weights/best.pt  # webcam
"""

import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(
        description="Microplastic Detection - Run inference on images, videos, or camera feed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Input source: image path, video path, directory, or camera index (0, 1, ...)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="results/weights/best.pt",
        help="Path to trained model weights (.pt file)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for Non-Maximum Suppression (NMS)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (pixels)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save detection results to disk",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="output/",
        help="Directory to save results",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display results in a window (not available on headless systems)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Device to run inference on: 'cpu', '0' (GPU), or '' (auto)",
    )
    return parser.parse_args()


def validate_inputs(args):
    """Validate that required files and paths exist."""
    # Check weights file
    if not Path(args.weights).exists():
        print(f"[ERROR] Model weights not found: {args.weights}")
        print("  → Download weights or specify correct path with --weights")
        sys.exit(1)

    # Check source (skip validation for camera index)
    if not args.source.isdigit() and not Path(args.source).exists():
        print(f"[ERROR] Source not found: {args.source}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)


def run_inference(args):
    """Load model and run detection inference."""
    print("=" * 60)
    print("  🔬 Microplastic Detection System")
    print("=" * 60)
    print(f"  Model:      {args.weights}")
    print(f"  Source:      {args.source}")
    print(f"  Confidence:  {args.conf}")
    print(f"  Image Size:  {args.imgsz}px")
    print(f"  Device:      {args.device if args.device else 'auto'}")
    print("=" * 60)

    # Load model
    print("\n[INFO] Loading model...")
    model = YOLO(args.weights)

    # Run inference
    print("[INFO] Running inference...")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=args.save,
        project=args.save_dir,
        name="detect",
        show=args.show,
        device=args.device if args.device else None,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("  📊 Detection Summary")
    print("=" * 60)

    total_detections = 0
    for i, result in enumerate(results):
        n_detections = len(result.boxes)
        total_detections += n_detections
        if hasattr(result, "path"):
            print(f"  Image {i + 1}: {Path(result.path).name} → {n_detections} microplastic(s) detected")

    print(f"\n  Total detections: {total_detections}")
    if args.save:
        print(f"  Results saved to: {args.save_dir}/detect/")
    print("=" * 60)

    return results


def main():
    """Main entry point for microplastic detection inference."""
    args = parse_args()
    validate_inputs(args)
    run_inference(args)


if __name__ == "__main__":
    main()
