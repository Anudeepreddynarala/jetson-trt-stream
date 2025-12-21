# ArcFace Recognition System

Real-time face recognition system using ArcFace with YOLO11 face detection.

## Prerequisites

- Docker installed on your system
- Webcam access

## Building the Docker Image

First, build the Docker image from the Dockerfile:

```bash
docker build -t arcface-recognition .
```

This creates an image based on **Python 3.10-slim** as the parent image, with all dependencies installed:
- OpenCV for video processing
- ONNX Runtime for ArcFace model inference
- Ultralytics for YOLO11 face detection
- Cryptography for secure embedding storage

## Running the Docker Container

Run the container with webcam and display access:

```bash
docker run -it --rm \
  --device /dev/video0:/dev/video0 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  arcface-recognition
```

## Usage

Once inside the container, you can perform two main operations:

### 1. Enroll a New Person

Add a new person to the face recognition database:

```bash
python enroll.py
```

This will:
- Access your webcam
- Detect your face using YOLO11
- Extract facial features using ArcFace
- Save the embedding to the encrypted database
- Prompt you to enter the person's name

### 2. Run Live Face Recognition

Identify faces in real-time from webcam:

```bash
python identify_live.py
```

This will:
- Open your webcam feed
- Detect faces in real-time
- Match detected faces against enrolled persons
- Display names and confidence scores
- Press 'q' to quit

## Files

- `enroll.py` - Enroll new faces into the system
- `identify_live.py` - Real-time face recognition
- `verify_live.py` - 1:1 face verification
- `arcface_infer.py` - ArcFace model inference
- `crypto_store.py` - Encrypted embedding storage
- `embeddings.json` - Encrypted face database
- `arcface.onnx` - ArcFace model
- `yolo11n.pt` - YOLO11 face detection model

## How It Works

1. **Face Detection**: YOLO11 detects faces in the video stream
2. **Feature Extraction**: ArcFace generates 512-dimensional embeddings
3. **Recognition**: Cosine similarity matching against enrolled faces
4. **Security**: All embeddings are encrypted at rest
