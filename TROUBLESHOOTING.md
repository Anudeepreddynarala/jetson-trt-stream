# Troubleshooting Guide

Common issues and solutions for YOLO-Rust-TRT.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Model Issues](#model-issues)
- [Runtime Issues](#runtime-issues)
- [Performance Issues](#performance-issues)
- [Network Issues](#network-issues)
- [Docker Issues](#docker-issues)

---

## Installation Issues

### Rust Not Found

**Problem**: `cargo: command not found`

**Solution**:
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify installation
cargo --version
```

### Python Dependencies Failed

**Problem**: `pip install` fails for some packages

**Solution**:
```bash
# Update pip first
pip3 install --upgrade pip

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3-dev python3-pip build-essential

# Install Python packages
pip3 install -r requirements.txt
```

### CUDA/TensorRT Not Available

**Problem**: TensorRT or CUDA libraries not found

**Solution**:
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Check TensorRT
dpkg -l | grep tensorrt

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

# For Jetson, TensorRT should be pre-installed with JetPack
```

---

## Model Issues

### Model File Not Found

**Problem**: `FileNotFoundError: models/yolov8n.trt`

**Solution**:
```bash
# Export and convert models
python3 scripts/export_model.py
python3 scripts/convert_to_trt.py

# Verify models exist
ls -lh models/
```

### ONNX Export Failed

**Problem**: `ultralytics` fails to export ONNX

**Solution**:
```bash
# Update ultralytics
pip3 install --upgrade ultralytics

# Try exporting with verbose output
python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', verbose=True)
"
```

### TensorRT Conversion Failed

**Problem**: `trtexec` fails with errors

**Solution**:
```bash
# Check TensorRT version
trtexec --help | head -5

# For TensorRT 10.x, use correct syntax:
trtexec --onnx=models/yolov8n.onnx \
        --saveEngine=models/yolov8n.trt \
        --fp16 \
        --memPoolSize=workspace:4096M

# If GPU memory is limited, reduce workspace:
trtexec --onnx=models/yolov8n.onnx \
        --saveEngine=models/yolov8n.trt \
        --fp16 \
        --memPoolSize=workspace:2048M
```

### PyCUDA Not Available

**Problem**: `WARNING: TensorRT/PyCUDA not available, will use ONNX Runtime fallback`

**Solution**:
```bash
# Install CUDA toolkit headers (if not installed)
sudo apt-get install cuda-toolkit-12-6

# Set CUDA environment
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install PyCUDA
pip3 install pycuda

# Note: ONNX Runtime fallback still works, just slower (CPU-only)
```

---

## Runtime Issues

### Camera Not Accessible

**Problem**: `Failed to open camera device 0`

**Solution**:
```bash
# Check camera devices
ls -la /dev/video*

# Test camera with v4l2
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --list-formats

# If using Docker, ensure device is mounted:
docker run --device /dev/video0 ...

# Check permissions
sudo usermod -aG video $USER
# Logout and login again
```

### Unix Socket Permission Denied

**Problem**: `Permission denied: /tmp/yolo_rust.sock`

**Solution**:
```bash
# Remove stale socket file
rm -f /tmp/yolo_rust.sock

# Check /tmp permissions
ls -ld /tmp

# If problem persists, use a different location in config.yaml:
networking:
  ipc_socket: "/home/$USER/yolo_rust.sock"
```

### Python Pipeline Crashes

**Problem**: Python process exits with segmentation fault

**Solution**:
```bash
# Run with debug output
python3 -u inference/main.py

# Check system logs
dmesg | tail -50

# Possible causes:
# 1. Out of memory - reduce camera resolution
# 2. CUDA driver issue - reboot system
# 3. Corrupted model - re-export and convert

# Try ONNX fallback (CPU mode):
# Edit inference/yolo_trt.py to force ONNX:
# self.backend = 'onnx'
```

### Rust Server Won't Start

**Problem**: Rust server fails to bind to port 8888

**Solution**:
```bash
# Check if port is already in use
sudo netstat -tlnp | grep 8888

# Kill existing process
sudo kill -9 <PID>

# Or use a different port in config.yaml:
networking:
  tcp_bind: "0.0.0.0:9888"

# Check if Unix socket exists and is in use
ls -la /tmp/yolo_rust.sock
sudo lsof /tmp/yolo_rust.sock

# Remove stale socket
rm -f /tmp/yolo_rust.sock
```

---

## Performance Issues

### Low FPS (< 10 FPS)

**Problem**: System running slower than expected

**Diagnosis**:
```bash
# Check if using TensorRT or ONNX fallback
# Look for: "Loading TensorRT engine" vs "Loading ONNX model"

# Monitor GPU usage
sudo tegrastats  # Jetson only
# or
nvidia-smi -l 1

# Check CPU usage
htop
```

**Solutions**:

1. **Using ONNX instead of TensorRT**:
   ```bash
   # Ensure TensorRT engine is built
   python3 scripts/convert_to_trt.py

   # Install PyCUDA for TensorRT support
   pip3 install pycuda
   ```

2. **GPU not being utilized**:
   ```bash
   # Check CUDA availability in Python
   python3 -c "import torch; print(torch.cuda.is_available())"

   # Verify TensorRT runtime
   python3 -c "import tensorrt; print(tensorrt.__version__)"
   ```

3. **Reduce camera resolution**:
   ```yaml
   # config/config.yaml
   camera:
     width: 640
     height: 480
   ```

4. **Use smaller YOLO model** (already using YOLOv8n, smallest)

### High Latency (> 100ms)

**Problem**: End-to-end latency too high

**Solutions**:

1. **Profile the pipeline**:
   ```python
   # Check metrics in Python output
   # Breakdown: preprocessing + inference + postprocessing
   ```

2. **Optimize preprocessing**:
   ```yaml
   # Reduce input resolution
   inference:
     input_size: 416  # down from 640
   ```

3. **Reduce buffer sizes**:
   ```yaml
   networking:
     buffer_size: 100  # down from 1000
   ```

4. **Check system load**:
   ```bash
   # Close other applications
   # Disable swap if using it
   sudo swapoff -a
   ```

### Memory Issues

**Problem**: Out of memory errors

**Solutions**:

1. **Monitor memory**:
   ```bash
   # Jetson
   sudo tegrastats

   # General
   free -h
   ```

2. **Reduce memory usage**:
   ```yaml
   camera:
     width: 640
     height: 480

   metrics:
     fps_window: 10  # down from 30
   ```

3. **Clear model cache**:
   ```bash
   # Remove and regenerate TensorRT engine
   rm models/yolov8n.trt
   python3 scripts/convert_to_trt.py
   ```

---

## Network Issues

### UDP Packets Not Received

**Problem**: `udp_test_client.py` receives no data

**Diagnosis**:
```bash
# Check if Rust server is running
ps aux | grep yolo-rust-networking

# Check UDP port
sudo netstat -ulnp | grep 9999

# Monitor network traffic
sudo tcpdump -i lo -n udp port 9999
```

**Solutions**:

1. **Verify Rust server is sending**:
   ```bash
   # Check Rust logs
   tail -f logs/rust_networking.log
   ```

2. **Check firewall**:
   ```bash
   sudo ufw status
   sudo ufw allow 9999/udp
   ```

3. **Test with netcat**:
   ```bash
   # In one terminal, start listener
   nc -ul 9999

   # In another, send test data
   echo "test" | nc -u 127.0.0.1 9999
   ```

### TCP Connection Refused

**Problem**: `nc 127.0.0.1 8888` fails

**Solutions**:
```bash
# Check if Rust server is listening
sudo netstat -tlnp | grep 8888

# Verify Rust server is running
ps aux | grep yolo-rust

# Check logs
tail -f logs/rust_networking.log

# Restart Rust server
pkill yolo-rust-networking
./scripts/run.sh --rust-only
```

---

## Docker Issues

### Build Failed

**Problem**: Docker build fails

**Solutions**:
```bash
# Clean Docker cache
docker system prune -a

# Build with no cache
docker build --no-cache -t yolo-rust-trt -f docker/Dockerfile .

# Check base image availability
docker pull nvcr.io/nvidia/l4t-tensorrt:r10.3-runtime

# If on Jetson, ensure sufficient disk space
df -h
```

### Container Crashes

**Problem**: Container exits immediately

**Solutions**:
```bash
# Check logs
docker logs <container-id>

# Run interactively
docker run -it --rm \
  --runtime nvidia \
  --device /dev/video0 \
  yolo-rust-trt /bin/bash

# Inside container, test components:
python3 scripts/test_inference.py
./rust_networking/target/release/yolo-rust-networking
```

### GPU Not Available in Container

**Problem**: CUDA not accessible inside Docker

**Solutions**:
```bash
# Ensure nvidia-docker runtime is installed
sudo apt-get install nvidia-docker2
sudo systemctl restart docker

# Use --runtime nvidia flag
docker run --runtime nvidia ...

# Verify GPU in container
docker run --runtime nvidia nvcr.io/nvidia/l4t-base:r35.1.0 nvidia-smi
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check logs**:
   ```bash
   tail -f logs/rust_networking.log
   tail -f logs/python_pipeline.log
   ```

2. **Run with verbose output**:
   ```bash
   python3 -u inference/main.py
   RUST_LOG=debug ./rust_networking/target/release/yolo-rust-networking
   ```

3. **Create a GitHub issue**:
   - Include system info (Jetson model, JetPack version)
   - Attach logs
   - Describe steps to reproduce

4. **Useful diagnostic commands**:
   ```bash
   # System info
   uname -a
   python3 --version
   cargo --version

   # GPU/CUDA info (Jetson)
   sudo tegrastats
   jetson_release

   # Python packages
   pip3 list | grep -E "ultralytics|onnx|tensorrt|torch"

   # Rust dependencies
   cargo tree
   ```
