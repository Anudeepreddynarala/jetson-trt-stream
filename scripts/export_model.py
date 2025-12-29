#!/usr/bin/env python3
"""
YOLO Model Export Script

Downloads YOLOv8n and exports to ONNX format, then converts to TensorRT engine.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO

def main():
    # Ensure models directory exists
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "yolov8n.pt"
    onnx_path = models_dir / "yolov8n.onnx"

    print("="*60)
    print("YOLO Model Export for TensorRT")
    print("="*60)

    # Step 1: Download or load YOLOv8n
    print("\n[1/2] Loading YOLOv8n model...")
    if not model_path.exists():
        print(f"  Downloading YOLOv8n to {model_path}")
        model = YOLO('yolov8n.pt')
        # Move downloaded model to models directory
        if Path('yolov8n.pt').exists():
            Path('yolov8n.pt').rename(model_path)
    else:
        print(f"  Loading existing model from {model_path}")
        model = YOLO(str(model_path))

    # Step 2: Export to ONNX
    print(f"\n[2/2] Exporting to ONNX format...")
    print(f"  Output: {onnx_path}")
    print("  Configuration:")
    print("    - Image size: 640x640")
    print("    - Dynamic shapes: False (required for TensorRT)")
    print("    - Simplify: True")
    print("    - Opset: 17")

    try:
        model.export(
            format='onnx',
            imgsz=640,
            dynamic=False,  # Static shapes for TensorRT
            simplify=True,   # Simplify ONNX graph
            opset=17         # ONNX opset version
        )

        # Move ONNX to models directory if needed
        if Path('yolov8n.onnx').exists() and not onnx_path.exists():
            Path('yolov8n.onnx').rename(onnx_path)

        print(f"\n✓ ONNX export successful: {onnx_path}")
        print(f"  Size: {onnx_path.stat().st_size / 1024 / 1024:.2f} MB")

    except Exception as e:
        print(f"\n✗ ONNX export failed: {e}")
        return 1

    # Step 3: Instructions for TensorRT conversion
    print("\n" + "="*60)
    print("Next Steps: Convert ONNX to TensorRT")
    print("="*60)
    print("\nRun the following command to create TensorRT engine:")
    print(f"""
/usr/src/tensorrt/bin/trtexec \\
    --onnx={onnx_path} \\
    --saveEngine={models_dir}/yolov8n.trt \\
    --fp16 \\
    --workspace=4096 \\
    --verbose
""")

    print("\nOr use the conversion script:")
    print(f"  python3 scripts/convert_to_trt.py")

    print("\n" + "="*60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
