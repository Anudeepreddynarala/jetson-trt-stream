_README content drafted with assistance from Claude_

# YOLO-Rust-TRT

GPU-accelerated YOLO object detection with Rust networking on NVIDIA Jetson Orin Nano.

## Overview

High-performance real-time object detection system combining:
- **Python + TensorRT**: Optimized YOLO inference (FP16 precision)
- **Rust**: Low-latency UDP/TCP networking layer
- **Unix Sockets**: Efficient inter-process communication

### Architecture

```
┌─────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
│ Camera  │────▶│ Python YOLO  │────▶│ Unix Socket │────▶│   Rust   │
│ (V4L2)  │     │  TensorRT    │     │   (IPC)     │     │ Networking│
└─────────┘     └──────────────┘     └─────────────┘     └─────┬────┘
                                                                │
                                                      ┌─────────┴──────────┐
                                                      │                    │
                                                   ┌──▼───┐          ┌────▼────┐
                                                   │ UDP  │          │   TCP   │
                                                   │:9999 │          │  :8888  │
                                                   └──────┘          └─────────┘
                                            (Detections Stream)  (Control API)
```

### Performance

**Target (Jetson Orin Nano)**:
- **FPS**: 25-30 frames/second
- **Latency**: <50ms end-to-end
- **Model**: YOLOv8n (6.3 MB)
- **Precision**: FP16 TensorRT

**Measured (TensorRT Benchmark)**:
- Throughput: 248.7 qps
- Inference: 4.0 ms (GPU compute)
- Total latency: 4.6 ms (with H2D/D2H transfers)

## Quick Start

### Prerequisites

- NVIDIA Jetson Orin Nano (JetPack 6.1+)
- Python 3.10+
- Rust 1.70+ (installed via `rustup`)
- TensorRT 10.3+
- CUDA 12.6+

### Installation

```bash
# Clone repository
git clone https://github.com/Anudeepreddynarala/yolo-rust-trt.git
cd yolo-rust-trt

# Install Python dependencies
pip3 install -r requirements.txt

# Install Rust (if not already installed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Build Rust networking layer
cd rust_networking
cargo build --release
cd ..
```

### Export and Convert Models

```bash
# Download and export YOLO to ONNX
python3 scripts/export_model.py

# Convert ONNX to TensorRT engine (FP16)
python3 scripts/convert_to_trt.py
```

This will create:
- `models/yolov8n.pt` - PyTorch model (6.3 MB)
- `models/yolov8n.onnx` - ONNX model (13 MB)
- `models/yolov8n.trt` - TensorRT engine (8.7 MB, FP16)

### Run

```bash
# Start both Rust server and Python pipeline
./scripts/run.sh

# Start only Rust networking layer
./scripts/run.sh --rust-only

# Start only Python inference pipeline
./scripts/run.sh --python-only
```

### Test

```bash
# Test YOLO inference (standalone)
python3 scripts/test_inference.py

# Monitor UDP detections stream
python3 scripts/udp_test_client.py

# Test TCP commands
echo '{"command":"STATS"}' | nc 127.0.0.1 8888
```

## Configuration

Edit `config/config.yaml`:

```yaml
camera:
  device_id: 0          # Camera device (/dev/video0)
  width: 1280           # Capture width
  height: 720           # Capture height
  fps: 30               # Target FPS

inference:
  model_path: "models/yolov8n.trt"  # TensorRT engine path
  input_size: 640                    # YOLO input size
  confidence_threshold: 0.5          # Detection threshold
  nms_threshold: 0.4                 # NMS IoU threshold

networking:
  ipc_socket: "/tmp/yolo_rust.sock"  # Unix socket path
  udp_target: "127.0.0.1:9999"       # UDP destination
  tcp_bind: "0.0.0.0:8888"           # TCP control port
  buffer_size: 1000                  # IPC buffer size

metrics:
  fps_window: 30                     # FPS averaging window
  report_interval_sec: 1.0           # Stats print interval
  enable_logging: true               # Enable file logging
  log_directory: "logs/"             # Log output directory
```

## TCP Control API

Connect to TCP port **8888** for runtime control.

### Commands

#### 1. Get Statistics
```bash
echo '{"command":"STATS"}' | nc 127.0.0.1 8888
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "cpu_usage_percent": 16.02,
    "memory_total_mb": 7619,
    "memory_used_mb": 2619,
    "packets_sent": 1234,
    "uptime_seconds": 120
  }
}
```

#### 2. Get Configuration
```bash
echo '{"command":"GET_CONFIG"}' | nc 127.0.0.1 8888
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "ipc_socket": "/tmp/yolo_rust.sock",
    "tcp_bind": "0.0.0.0:8888",
    "udp_target": "127.0.0.1:9999"
  }
}
```

#### 3. Start/Stop (State Control)
```bash
echo '{"command":"START"}' | nc 127.0.0.1 8888
echo '{"command":"STOP"}' | nc 127.0.0.1 8888
```

**Response:**
```json
{
  "status": "success",
  "data": {
    "state": "running"  // or "stopped"
  }
}
```

### Error Response
```json
{
  "status": "error",
  "error": "Error message here"
}
```

## UDP Detection Stream

Detections are streamed via UDP to `127.0.0.1:9999` in JSON format.

### Message Format

```json
{
  "message_type": "detection",
  "timestamp_ns": 1735509432123456789,
  "frame_id": 42,
  "detections": [
    {
      "label": "person",
      "confidence": 0.92,
      "bbox": {
        "x1": 120.5,
        "y1": 80.3,
        "x2": 340.2,
        "y2": 450.8
      }
    }
  ],
  "metrics": {
    "preprocessing_ms": 3.2,
    "inference_ms": 8.1,
    "postprocessing_ms": 4.5,
    "fps": 28.3,
    "total_latency_ms": 35.7
  }
}
```

## Project Structure

```
yolo-rust-trt/
├── config/
│   └── config.yaml              # Main configuration
├── docker/
│   ├── Dockerfile               # Multi-stage build
│   ├── docker-compose.yml       # Docker Compose setup
│   └── entrypoint.sh            # Container startup script
├── inference/
│   ├── camera.py                # Camera capture module
│   ├── ipc_client.py            # Unix socket IPC client
│   ├── main.py                  # Main inference pipeline
│   ├── metrics.py               # Performance metrics
│   └── yolo_trt.py              # YOLO TensorRT inference
├── logs/                        # Runtime logs
├── models/                      # YOLO models
│   ├── yolov8n.pt               # PyTorch model
│   ├── yolov8n.onnx             # ONNX model
│   └── yolov8n.trt              # TensorRT engine
├── rust_networking/
│   ├── Cargo.toml               # Rust dependencies
│   └── src/
│       ├── main.rs              # Main entry point
│       ├── protocol.rs          # Message schemas
│       ├── tcp.rs               # TCP command server
│       └── udp.rs               # UDP streamer
├── scripts/
│   ├── convert_to_trt.py        # ONNX → TensorRT converter
│   ├── export_model.py          # PyTorch → ONNX exporter
│   ├── run.sh                   # Launch script
│   ├── test_inference.py        # Inference test
│   └── udp_test_client.py       # UDP receiver test
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## COCO Classes

YOLOv8n detects 80 COCO classes including: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, and more.

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

## License

MIT License

## Credits

**Author**: Anudeep Reddy Narala
**GitHub**: https://github.com/Anudeepreddynarala
**Built With**: Ultralytics YOLOv8, NVIDIA TensorRT, Rust, Tokio
