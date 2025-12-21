# Use minimal Python base image
FROM python:3.10-slim

# Install only essential runtime libraries (no dev packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python packages (using headless OpenCV to save space)
RUN pip install --no-cache-dir \
    opencv-python-headless==4.10.0.84 \
    onnxruntime==1.20.1 \
    ultralytics==8.3.41 \
    numpy==1.26.4 \
    cryptography==44.0.0

# Copy only necessary files (not .claude, .git, etc.)
COPY *.py *.onnx *.pt /app/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command - open interactive shell
CMD ["/bin/bash"]
