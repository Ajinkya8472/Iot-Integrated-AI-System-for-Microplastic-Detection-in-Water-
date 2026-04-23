"""
Microplastic Detection - Raspberry Pi Camera Capture & Inference
=================================================================
Real-time microplastic detection using Raspberry Pi Camera Module or USB camera.
Captures frames, runs YOLO inference, and displays/saves detection results.

Usage (on Raspberry Pi):
    python src/capture.py --weights results/weights/best.pt
    python src/capture.py --weights results/weights/best.pt --camera 0 --save-frames

Usage (desktop with webcam):
    python src/capture.py --weights results/weights/best.pt --camera 0 --no-pi
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Real-time microplastic detection using camera + YOLO model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="results/weights/best.pt",
        help="Path to trained YOLO model weights",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index (0 for default camera)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detection",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (pixels)",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        default=False,
        help="Save frames with detections to disk",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/captures",
        help="Directory to save captured frames",
    )
    parser.add_argument(
        "--no-pi",
        action="store_true",
        default=False,
        help="Use OpenCV VideoCapture instead of PiCamera (for desktop testing)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[640, 480],
        help="Camera resolution (width height)",
    )
    return parser.parse_args()


def init_camera_opencv(camera_index, resolution):
    """Initialize camera using OpenCV VideoCapture."""
    print(f"[INFO] Initializing camera (OpenCV, index={camera_index})...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("[ERROR] Could not open camera!")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    print(f"[INFO] Camera ready: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    return cap


def init_camera_picamera(resolution):
    """Initialize camera using PiCamera2 (Raspberry Pi only)."""
    try:
        from picamera2 import Picamera2

        print("[INFO] Initializing PiCamera2...")
        picam = Picamera2()
        config = picam.create_preview_configuration(
            main={"size": tuple(resolution), "format": "RGB888"}
        )
        picam.configure(config)
        picam.start()
        time.sleep(2)  # Allow camera warm-up
        print(f"[INFO] PiCamera ready: {resolution[0]}x{resolution[1]}")
        return picam
    except ImportError:
        print("[ERROR] picamera2 not installed. Install with: pip install picamera2")
        print("  → Or use --no-pi flag for OpenCV camera")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] PiCamera initialization failed: {e}")
        sys.exit(1)


def draw_detection_overlay(frame, results, fps):
    """Draw detection boxes and info overlay on the frame."""
    annotated = results[0].plot() if results else frame

    # FPS counter
    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    # Detection count
    n_detections = len(results[0].boxes) if results else 0
    color = (0, 0, 255) if n_detections > 0 else (0, 255, 0)
    status = f"DETECTED: {n_detections}" if n_detections > 0 else "CLEAR"
    cv2.putText(
        annotated,
        status,
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
    )

    # Title bar
    cv2.putText(
        annotated,
        "Microplastic Detection System",
        (10, annotated.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
    )

    return annotated, n_detections


def run_capture_loop(args):
    """Main capture and inference loop."""
    print("=" * 60)
    print("  🔬 Microplastic Detection - Live Camera Feed")
    print("=" * 60)
    print(f"  Model:       {args.weights}")
    print(f"  Confidence:  {args.conf}")
    print(f"  Resolution:  {args.resolution[0]}x{args.resolution[1]}")
    print(f"  Camera Mode: {'OpenCV' if args.no_pi else 'PiCamera2'}")
    print("=" * 60)

    # Load model
    print("\n[INFO] Loading YOLO model...")
    model = YOLO(args.weights)

    # Initialize camera
    use_opencv = args.no_pi
    if use_opencv:
        camera = init_camera_opencv(args.camera, args.resolution)
    else:
        camera = init_camera_picamera(args.resolution)

    # Create output directory
    if args.save_frames:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[INFO] Saving frames to: {args.output_dir}")

    print("\n[INFO] Starting detection loop... Press 'q' to quit, 's' to save screenshot\n")

    frame_count = 0
    detection_count = 0
    fps = 0

    try:
        while True:
            t_start = time.time()

            # Capture frame
            if use_opencv:
                ret, frame = camera.read()
                if not ret:
                    print("[ERROR] Failed to capture frame")
                    break
            else:
                frame = camera.capture_array()

            # Run inference
            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )

            # Draw overlay
            annotated, n_det = draw_detection_overlay(frame, results, fps)
            frame_count += 1
            detection_count += n_det

            # Auto-save frames with detections
            if args.save_frames and n_det > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                save_path = os.path.join(args.output_dir, f"detection_{timestamp}.jpg")
                cv2.imwrite(save_path, annotated)

            # Display
            cv2.imshow("Microplastic Detection", annotated)

            # Calculate FPS
            fps = 1.0 / (time.time() - t_start)

            # Key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(args.output_dir, f"screenshot_{timestamp}.jpg")
                os.makedirs(args.output_dir, exist_ok=True)
                cv2.imwrite(save_path, annotated)
                print(f"[INFO] Screenshot saved: {save_path}")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        # Cleanup
        if use_opencv:
            camera.release()
        else:
            camera.stop()
        cv2.destroyAllWindows()

        # Print summary
        print("\n" + "=" * 60)
        print("  📊 Session Summary")
        print("=" * 60)
        print(f"  Total frames processed:  {frame_count}")
        print(f"  Total detections:        {detection_count}")
        print(f"  Average FPS:             {fps:.1f}")
        print("=" * 60)


def main():
    """Main entry point for camera capture & detection."""
    args = parse_args()

    # Validate weights
    if not Path(args.weights).exists():
        print(f"[ERROR] Model weights not found: {args.weights}")
        sys.exit(1)

    run_capture_loop(args)


if __name__ == "__main__":
    main()
