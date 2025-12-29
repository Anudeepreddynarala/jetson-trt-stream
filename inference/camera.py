#!/usr/bin/env python3
"""
Camera Capture Module

Handles video capture from camera with error handling and configuration.
"""

import cv2
import time
from pathlib import Path


class Camera:
    """Camera capture wrapper with error handling"""

    def __init__(self, device_id=0, width=1280, height=720, fps=30):
        """
        Initialize camera

        Args:
            device_id: Camera device ID (default: 0)
            width: Frame width
            height: Frame height
            fps: Target FPS
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps

        self.cap = None
        self.frame_count = 0
        self.dropped_frames = 0

    def open(self):
        """Open camera connection"""
        print(f"Opening camera {self.device_id}...")

        self.cap = cv2.VideoCapture(self.device_id)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {self.device_id}")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Get actual properties
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        print(f"✓ Camera opened:")
        print(f"  Resolution: {actual_width}x{actual_height}")
        print(f"  FPS: {actual_fps}")

        return self

    def read(self):
        """
        Read frame from camera

        Returns:
            frame: BGR image or None if failed
            timestamp_ns: Capture timestamp in nanoseconds
        """
        if self.cap is None:
            raise RuntimeError("Camera not opened. Call open() first")

        timestamp_ns = time.time_ns()
        ret, frame = self.cap.read()

        if not ret:
            self.dropped_frames += 1
            return None, timestamp_ns

        self.frame_count += 1
        return frame, timestamp_ns

    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

            print(f"\nCamera stats:")
            print(f"  Total frames: {self.frame_count}")
            print(f"  Dropped frames: {self.dropped_frames}")

            if self.frame_count > 0:
                drop_rate = (self.dropped_frames / (self.frame_count + self.dropped_frames)) * 100
                print(f"  Drop rate: {drop_rate:.2f}%")

    def __enter__(self):
        """Context manager support"""
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.release()


if __name__ == "__main__":
    # Test camera capture
    print("Testing camera capture...")
    print("Press Ctrl+C to stop\n")

    try:
        with Camera(device_id=0) as cam:
            start_time = time.time()
            frames_captured = 0

            while True:
                frame, timestamp = cam.read()

                if frame is None:
                    print("Warning: Failed to capture frame")
                    continue

                frames_captured += 1

                # Display stats every second
                elapsed = time.time() - start_time
                if elapsed >= 1.0:
                    fps = frames_captured / elapsed
                    print(f"FPS: {fps:.1f} | Resolution: {frame.shape[1]}x{frame.shape[0]} | Frames: {frames_captured}")
                    start_time = time.time()
                    frames_captured = 0

                # Small delay to prevent busy loop
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nCamera test stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
