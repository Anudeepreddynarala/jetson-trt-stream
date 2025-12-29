#!/usr/bin/env python3
"""
Main Inference Pipeline

Integrates camera capture, YOLO inference, and IPC communication.
"""

import sys
import time
import signal
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

from inference.yolo_trt import YOLOTensorRT
from inference.camera import Camera
from inference.metrics import MetricsCollector
from inference.ipc_client import IPCClient

import yaml


class InferencePipeline:
    """Main inference pipeline orchestrator"""

    def __init__(self, config_path="config/config.yaml"):
        """
        Initialize inference pipeline

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        config_file = Path(__file__).parent.parent / config_path
        with open(config_file) as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.yolo = None
        self.camera = None
        self.metrics = None
        self.ipc = None

        self.running = False
        self.frame_id = 0

    def setup(self):
        """Setup all components"""
        print("="*60)
        print("YOLO-TensorRT-Rust Inference Pipeline")
        print("="*60)
        print()

        # Initialize YOLO engine
        model_path = self.config['inference']['model_path']
        conf_threshold = self.config['inference']['confidence_threshold']

        print("Loading YOLO model...")
        self.yolo = YOLOTensorRT(
            engine_path=model_path,
            conf_threshold=conf_threshold
        )
        print()

        # Initialize camera
        cam_config = self.config['camera']
        self.camera = Camera(
            device_id=cam_config['device_id'],
            width=cam_config.get('width', 1280),
            height=cam_config.get('height', 720),
            fps=cam_config.get('fps', 30)
        )
        self.camera.open()
        print()

        # Initialize metrics
        metrics_config = self.config['metrics']
        self.metrics = MetricsCollector(
            window_size=metrics_config.get('fps_window', 30),
            report_interval=metrics_config.get('report_interval_sec', 1.0)
        )

        # Initialize IPC (optional - may not have Rust server yet)
        ipc_socket = self.config['networking']['ipc_socket']
        self.ipc = IPCClient(socket_path=ipc_socket)

        # Try to connect (don't fail if Rust server not running)
        print("Attempting to connect to Rust server...")
        if self.ipc.connect(max_retries=3, retry_delay=0.5):
            print("✓ Connected to Rust networking layer\n")
        else:
            print("⚠ Rust server not available - running in standalone mode\n")

        print("="*60)
        print("Pipeline ready! Press Ctrl+C to stop")
        print("="*60)
        print()

    def process_frame(self, frame, timestamp_ns):
        """
        Process single frame through pipeline

        Args:
            frame: Camera frame
            timestamp_ns: Capture timestamp

        Returns:
            detections: List of detected objects
        """
        # Run inference
        detections, inference_metrics = self.yolo(frame)

        # Build message
        message = {
            "message_type": "detection",
            "timestamp_ns": timestamp_ns,
            "frame_id": self.frame_id,
            "detections": detections,
            "metrics": {
                "preprocessing_ms": inference_metrics['preprocessing_ms'],
                "inference_ms": inference_metrics['inference_ms'],
                "postprocessing_ms": inference_metrics['postprocessing_ms']
            }
        }

        # Send to Rust (if connected)
        if self.ipc and self.ipc.connected:
            self.ipc.send_detection(message)

        # Record metrics
        self.metrics.record_frame(
            frame_latency_ms=inference_metrics['total_ms'],
            metrics={
                'capture_ms': 0,  # Camera capture time not tracked separately
                'inference_ms': inference_metrics['inference_ms'],
                'postprocessing_ms': inference_metrics['postprocessing_ms']
            }
        )

        return detections

    def run(self):
        """Main processing loop"""
        self.running = True

        try:
            while self.running:
                # Capture frame
                frame, timestamp_ns = self.camera.read()

                if frame is None:
                    print("Warning: Failed to capture frame")
                    continue

                # Process frame
                detections = self.process_frame(frame, timestamp_ns)

                # Update frame counter
                self.frame_id += 1

                # Print stats periodically
                if self.metrics.should_report():
                    self.metrics.print_stats(detections_count=len(detections))

        except KeyboardInterrupt:
            print("\n\nShutting down...")

    def cleanup(self):
        """Cleanup resources"""
        if self.camera:
            self.camera.release()

        if self.ipc:
            self.ipc.close()

        print("\n✓ Pipeline shutdown complete")

    def __enter__(self):
        """Context manager support"""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.cleanup()


def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nReceived interrupt signal")
    sys.exit(0)


if __name__ == "__main__":
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    try:
        with InferencePipeline() as pipeline:
            pipeline.run()

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
