# Stage 1: Rust build
FROM rust:1.75 AS rust-builder
WORKDIR /build
COPY rust_networking/ .
RUN cargo build --release

# Stage 2: Python runtime with TensorRT
FROM nvcr.io/nvidia/l4t-tensorrt:r10.3-runtime

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy Rust binary from builder
COPY --from=rust-builder /build/target/release/yolo-rust-networking /app/rust_server

# Copy Python code
COPY inference/ /app/inference/
COPY config/ /app/config/
COPY scripts/ /app/scripts/

# Create models directory
RUN mkdir -p /app/models /app/logs

# Expose ports
EXPOSE 8888/tcp
EXPOSE 9999/udp

# Launch script
COPY docker/entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
