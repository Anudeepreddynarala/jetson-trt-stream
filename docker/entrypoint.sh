#!/bin/bash
set -e

echo "Starting YOLO-TensorRT-Rust System..."

# Start Rust networking in background
/app/rust_server &
RUST_PID=$!

echo "Rust networking started (PID: $RUST_PID)"

# Wait for Rust to initialize
sleep 2

# Start Python inference
cd /app && python3 inference/yolo_trt.py

# Cleanup on exit
kill $RUST_PID 2>/dev/null || true
