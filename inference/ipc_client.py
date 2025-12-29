#!/usr/bin/env python3
"""
IPC Client for Rust Communication

Handles Unix socket communication between Python inference and Rust networking layer.
"""

import socket
import json
import time
from pathlib import Path


class IPCClient:
    """Unix socket client for sending detections to Rust server"""

    def __init__(self, socket_path="/tmp/yolo_rust.sock"):
        """
        Initialize IPC client

        Args:
            socket_path: Path to Unix domain socket
        """
        self.socket_path = socket_path
        self.sock = None
        self.connected = False
        self.messages_sent = 0
        self.reconnect_attempts = 0

    def connect(self, max_retries=5, retry_delay=0.5):
        """
        Connect to Rust server

        Args:
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retries in seconds
        """
        print(f"Connecting to Rust server at {self.socket_path}...")

        for attempt in range(max_retries):
            try:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.connect(self.socket_path)
                self.connected = True
                print(f"✓ Connected to Rust server")
                return True

            except (FileNotFoundError, ConnectionRefusedError) as e:
                if attempt < max_retries - 1:
                    print(f"  Attempt {attempt + 1}/{max_retries} failed, retrying...")
                    time.sleep(retry_delay)
                else:
                    print(f"✗ Failed to connect after {max_retries} attempts")
                    print(f"  Make sure Rust server is running")
                    return False

        return False

    def send_detection(self, detection_data):
        """
        Send detection message to Rust server

        Args:
            detection_data: Dict containing detection information

        Returns:
            bool: True if sent successfully, False otherwise
        """
        if not self.connected:
            print("Warning: Not connected to Rust server")
            return False

        try:
            # Serialize to JSON with newline delimiter
            message = json.dumps(detection_data) + "\n"

            # Send message
            self.sock.sendall(message.encode('utf-8'))
            self.messages_sent += 1
            return True

        except BrokenPipeError:
            print("Error: Connection to Rust server lost")
            self.connected = False
            return False

        except Exception as e:
            print(f"Error sending message: {e}")
            return False

    def reconnect(self):
        """Attempt to reconnect to server"""
        print("Attempting to reconnect...")
        self.reconnect_attempts += 1

        if self.sock:
            try:
                self.sock.close()
            except:
                pass

        self.sock = None
        self.connected = False

        return self.connect(max_retries=3, retry_delay=1.0)

    def send_with_retry(self, data, max_retries=3):
        """
        Send data with automatic retry on failure

        Args:
            data: Data to send
            max_retries: Maximum retry attempts

        Returns:
            bool: True if sent successfully
        """
        for attempt in range(max_retries):
            if self.send_detection(data):
                return True

            # Try to reconnect if send failed
            if attempt < max_retries - 1:
                if not self.reconnect():
                    return False

        return False

    def close(self):
        """Close connection"""
        if self.sock:
            try:
                self.sock.close()
            except:
                pass

            self.sock = None
            self.connected = False

            print(f"\nIPC Stats:")
            print(f"  Messages sent: {self.messages_sent}")
            print(f"  Reconnect attempts: {self.reconnect_attempts}")

    def __enter__(self):
        """Context manager support"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.close()


if __name__ == "__main__":
    # Test IPC client
    print("Testing IPC client...")
    print("Note: This test will fail if Rust server is not running\n")

    # Create test detection message
    test_message = {
        "message_type": "detection",
        "timestamp_ns": time.time_ns(),
        "frame_id": 123,
        "detections": [
            {
                "class_id": 0,
                "class_name": "person",
                "confidence": 0.92,
                "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 500}
            }
        ],
        "metrics": {
            "capture_ms": 2.3,
            "inference_ms": 8.5,
            "postprocessing_ms": 1.2
        }
    }

    try:
        with IPCClient() as ipc:
            if ipc.connected:
                print("Sending test message...")
                success = ipc.send_detection(test_message)

                if success:
                    print("✓ Test message sent successfully")
                else:
                    print("✗ Failed to send test message")
            else:
                print("✗ Could not connect to server")
                print("  Start Rust server first with:")
                print("  cd rust_networking && cargo run")

    except Exception as e:
        print(f"Error: {e}")
