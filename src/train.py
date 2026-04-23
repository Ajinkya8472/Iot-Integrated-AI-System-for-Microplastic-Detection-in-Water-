"""
Microplastic Detection - Training Script
==========================================
Train a YOLO model for microplastic detection using custom dataset.

Usage:
    python src/train.py --data configs/data.yaml --epochs 50
    python src/train.py --data configs/data.yaml --model yolov5nu.pt --epochs 100 --batch 16
    python src/train.py --data configs/data.yaml --model yolo11n.pt --epochs 80 --imgsz 640
"""

import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Microplastic Detection - Train a YOLO model on microplastic dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        default="configs/data.yaml",
        help="Path to dataset YAML configuration",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov5nu.pt",
        help="Pre-trained model to fine-tune (e.g., yolov5nu.pt, yolo11n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size for training",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Training image size (pixels)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Training device: 'cpu', '0' (GPU), '0,1' (multi-GPU), or '' (auto)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="results",
        help="Project directory for saving training outputs",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="train",
        help="Experiment name",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="auto",
        choices=["SGD", "Adam", "AdamW", "auto"],
        help="Optimizer for training",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from last checkpoint",
    )
    return parser.parse_args()


def validate_inputs(args):
    """Validate that required files exist before training."""
    # Check data config
    if not Path(args.data).exists():
        print(f"[ERROR] Dataset config not found: {args.data}")
        print("  → Create configs/data.yaml or specify path with --data")
        sys.exit(1)

    # Create project directory
    os.makedirs(args.project, exist_ok=True)

    print("[INFO] Input validation passed ✓")


def run_training(args):
    """Load model and start training."""
    print("=" * 60)
    print("  🔬 Microplastic Detection - Model Training")
    print("=" * 60)
    print(f"  Model:       {args.model}")
    print(f"  Dataset:     {args.data}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch Size:  {args.batch}")
    print(f"  Image Size:  {args.imgsz}px")
    print(f"  Optimizer:   {args.optimizer}")
    print(f"  LR:          {args.lr0}")
    print(f"  Patience:    {args.patience}")
    print(f"  Device:      {args.device if args.device else 'auto'}")
    print(f"  Output:      {args.project}/{args.name}")
    print("=" * 60)

    # Load model
    print("\n[INFO] Loading pre-trained model...")
    model = YOLO(args.model)

    # Start training
    print("[INFO] Starting training...\n")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        patience=args.patience,
        optimizer=args.optimizer,
        lr0=args.lr0,
        resume=args.resume,
        pretrained=True,
        save=True,
        plots=True,
        verbose=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("  📊 Training Complete!")
    print("=" * 60)
    print(f"  Results saved to: {args.project}/{args.name}/")
    print(f"  Best weights:     {args.project}/{args.name}/weights/best.pt")
    print("=" * 60)

    return results


def main():
    """Main entry point for model training."""
    args = parse_args()
    validate_inputs(args)
    run_training(args)


if __name__ == "__main__":
    main()
