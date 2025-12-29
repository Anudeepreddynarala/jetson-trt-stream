#!/usr/bin/env python3
"""
Test YOLO Inference - Verify TensorRT model works
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.yolo_trt import YOLOTensorRT


def create_test_image(width=640, height=640):
    """Create a simple test image with colored rectangles"""
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Add some colored rectangles to simulate objects
    cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), -1)  # Green
    cv2.rectangle(img, (350, 150), (550, 400), (255, 0, 0), -1)  # Blue
    cv2.rectangle(img, (50, 400), (250, 550), (0, 0, 255), -1)   # Red

    # Add some text
    cv2.putText(img, "Test Image", (200, 50), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2)

    return img


def main():
    print("="*60)
    print("YOLO TensorRT Inference Test")
    print("="*60)
    print()

    # Initialize YOLO
    model_path = "models/yolov8n.trt"
    print(f"Loading model: {model_path}")

    try:
        yolo = YOLOTensorRT(
            engine_path=model_path,
            conf_threshold=0.5
        )
        print("✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nTrying ONNX model as fallback...")
        try:
            yolo = YOLOTensorRT(
                engine_path="models/yolov8n.onnx",
                conf_threshold=0.5
            )
            print("✓ ONNX model loaded successfully\n")
        except Exception as e2:
            print(f"✗ Failed to load ONNX model: {e2}")
            return 1

    # Create test image
    print("Creating test image...")
    test_img = create_test_image()
    print(f"✓ Test image created: {test_img.shape}\n")

    # Run inference multiple times to measure performance
    print("Running inference (10 iterations)...")
    print("-"*60)

    total_time = 0
    num_runs = 10

    for i in range(num_runs):
        start = time.perf_counter()
        detections, metrics = yolo(test_img)
        elapsed = (time.perf_counter() - start) * 1000
        total_time += elapsed

        print(f"Run {i+1:2d}: {elapsed:6.2f}ms | "
              f"Pre: {metrics['preprocessing_ms']:5.2f}ms | "
              f"Inf: {metrics['inference_ms']:5.2f}ms | "
              f"Post: {metrics['postprocessing_ms']:5.2f}ms | "
              f"Detections: {len(detections)}")

    avg_time = total_time / num_runs
    fps = 1000.0 / avg_time if avg_time > 0 else 0

    print("-"*60)
    print(f"\nPerformance Summary:")
    print(f"  Average latency: {avg_time:.2f}ms")
    print(f"  Throughput: {fps:.1f} FPS")
    print(f"  Total runs: {num_runs}")

    print("\n" + "="*60)
    print("✓ Inference test completed successfully")
    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
