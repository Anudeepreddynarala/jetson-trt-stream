#!/usr/bin/env python3
"""
UDP Test Client - Receives and displays detection messages from Rust UDP streamer
"""

import socket
import json
import sys
from datetime import datetime

def main():
    # UDP settings
    UDP_IP = "127.0.0.1"
    UDP_PORT = 9999

    # Create UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))

    print(f"[UDP Test Client] Listening on {UDP_IP}:{UDP_PORT}")
    print("Waiting for detection messages...\n")

    packet_count = 0

    try:
        while True:
            # Receive data
            data, addr = sock.recvfrom(65536)  # Max UDP packet size
            packet_count += 1

            try:
                # Parse JSON
                message = json.loads(data.decode('utf-8'))

                # Display detection info
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                num_detections = len(message.get('detections', []))
                fps = message.get('metrics', {}).get('fps', 0.0)
                latency = message.get('metrics', {}).get('total_latency_ms', 0.0)

                print(f"[{timestamp}] Packet #{packet_count} from {addr[0]}:{addr[1]}")
                print(f"  Detections: {num_detections}")
                print(f"  FPS: {fps:.1f}")
                print(f"  Latency: {latency:.1f}ms")

                # Show first 3 detections
                for i, det in enumerate(message.get('detections', [])[:3]):
                    label = det.get('label', 'unknown')
                    conf = det.get('confidence', 0.0)
                    bbox = det.get('bbox', {})
                    print(f"    [{i+1}] {label} ({conf:.2%}) - " +
                          f"[{bbox.get('x1',0):.0f},{bbox.get('y1',0):.0f}," +
                          f"{bbox.get('x2',0):.0f},{bbox.get('y2',0):.0f}]")

                if num_detections > 3:
                    print(f"    ... and {num_detections - 3} more")

                print()

            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse JSON: {e}")
            except Exception as e:
                print(f"[ERROR] {e}")

    except KeyboardInterrupt:
        print(f"\n\n[UDP Test Client] Received {packet_count} packets. Exiting...")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
