#!/usr/bin/env python3
"""
TensorRT Conversion Script

Converts ONNX model to TensorRT engine with FP16 precision.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    models_dir = Path(__file__).parent.parent / "models"
    onnx_path = models_dir / "yolov8n.onnx"
    trt_path = models_dir / "yolov8n.trt"

    print("="*60)
    print("TensorRT Engine Conversion")
    print("="*60)

    # Check if ONNX exists
    if not onnx_path.exists():
        print(f"\n✗ ONNX model not found: {onnx_path}")
        print("  Run 'python3 scripts/export_model.py' first")
        return 1

    # Check if trtexec exists
    trtexec_path = "/usr/src/tensorrt/bin/trtexec"
    if not Path(trtexec_path).exists():
        print(f"\n✗ trtexec not found at {trtexec_path}")
        print("  TensorRT may not be installed correctly")
        return 1

    print(f"\nInput:  {onnx_path}")
    print(f"Output: {trt_path}")
    print("\nConfiguration:")
    print("  - Precision: FP16")
    print("  - Workspace: 4096 MB")
    print("  - Input format: FP16:CHW")
    print("  - Output format: FP16:CHW")

    # Build command
    cmd = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={trt_path}",
        "--fp16",
        "--memPoolSize=workspace:4096M",
        "--verbose"
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    print("="*60)

    try:
        # Run trtexec
        result = subprocess.run(cmd, check=True, capture_output=False)

        if trt_path.exists():
            print("\n" + "="*60)
            print("✓ TensorRT engine created successfully!")
            print(f"  Path: {trt_path}")
            print(f"  Size: {trt_path.stat().st_size / 1024 / 1024:.2f} MB")
            print("="*60)
            return 0
        else:
            print("\n✗ Engine file not created")
            return 1

    except subprocess.CalledProcessError as e:
        print(f"\n✗ TensorRT conversion failed: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n\nConversion interrupted by user")
        return 130

if __name__ == "__main__":
    sys.exit(main())
