#!/usr/bin/env python3
"""
Performance Metrics Collection

Tracks FPS, latency, and system performance metrics.
"""

import time
from collections import deque
import os


class MetricsCollector:
    """Collects and reports performance metrics"""

    def __init__(self, window_size=30, report_interval=1.0):
        """
        Initialize metrics collector

        Args:
            window_size: Number of frames for sliding window FPS calculation
            report_interval: How often to print stats (seconds)
        """
        self.window_size = window_size
        self.report_interval = report_interval

        # Frame timing
        self.frame_times = deque(maxlen=window_size)
        self.latencies = deque(maxlen=window_size)

        # Counters
        self.total_frames = 0
        self.start_time = time.time()
        self.last_report_time = time.time()

        # Component timings
        self.capture_times = deque(maxlen=window_size)
        self.inference_times = deque(maxlen=window_size)
        self.postprocess_times = deque(maxlen=window_size)

    def record_frame(self, frame_latency_ms, metrics=None):
        """
        Record frame processing metrics

        Args:
            frame_latency_ms: Total frame processing time
            metrics: Optional dict with component timings
        """
        current_time = time.time()

        self.frame_times.append(current_time)
        self.latencies.append(frame_latency_ms)
        self.total_frames += 1

        if metrics:
            self.capture_times.append(metrics.get('capture_ms', 0))
            self.inference_times.append(metrics.get('inference_ms', 0))
            self.postprocess_times.append(metrics.get('postprocessing_ms', 0))

    def get_fps(self):
        """Calculate current FPS from sliding window"""
        if len(self.frame_times) < 2:
            return 0.0

        time_span = self.frame_times[-1] - self.frame_times[0]
        if time_span == 0:
            return 0.0

        return (len(self.frame_times) - 1) / time_span

    def get_avg_latency(self):
        """Get average latency in ms"""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    def should_report(self):
        """Check if it's time to print stats"""
        current_time = time.time()
        if current_time - self.last_report_time >= self.report_interval:
            self.last_report_time = current_time
            return True
        return False

    def get_summary(self):
        """Get current metrics summary"""
        uptime = time.time() - self.start_time

        summary = {
            'uptime_seconds': uptime,
            'total_frames': self.total_frames,
            'fps': self.get_fps(),
            'avg_latency_ms': self.get_avg_latency(),
        }

        # Component averages
        if self.capture_times:
            summary['avg_capture_ms'] = sum(self.capture_times) / len(self.capture_times)
        if self.inference_times:
            summary['avg_inference_ms'] = sum(self.inference_times) / len(self.inference_times)
        if self.postprocess_times:
            summary['avg_postprocess_ms'] = sum(self.postprocess_times) / len(self.postprocess_times)

        return summary

    def print_stats(self, detections_count=0):
        """Print formatted statistics"""
        summary = self.get_summary()

        # Clear screen (optional)
        # os.system('clear' if os.name == 'posix' else 'cls')

        print("=" * 60)
        print("YOLO-TensorRT-Rust Performance Metrics")
        print("=" * 60)
        print(f"Uptime:        {summary['uptime_seconds']:.1f}s")
        print(f"Total Frames:  {summary['total_frames']}")
        print(f"")
        print(f"FPS:           {summary['fps']:.1f} fps")
        print(f"Avg Latency:   {summary['avg_latency_ms']:.1f} ms")
        print(f"")

        if 'avg_capture_ms' in summary:
            print(f"[Timing Breakdown]")
            print(f"  Capture:     {summary.get('avg_capture_ms', 0):.1f} ms")
            print(f"  Inference:   {summary.get('avg_inference_ms', 0):.1f} ms")
            print(f"  Postprocess: {summary.get('avg_postprocess_ms', 0):.1f} ms")
            print(f"")

        print(f"[Last Frame]")
        print(f"  Detections:  {detections_count}")
        print("=" * 60)


if __name__ == "__main__":
    # Test metrics collector
    print("Testing metrics collector...\n")

    metrics = MetricsCollector(window_size=30, report_interval=1.0)

    # Simulate frame processing
    for i in range(100):
        # Simulate some work
        time.sleep(0.033)  # ~30 FPS

        # Record metrics
        metrics.record_frame(
            frame_latency_ms=33.0,
            metrics={
                'capture_ms': 2.0,
                'inference_ms': 25.0,
                'postprocessing_ms': 6.0
            }
        )

        # Print stats periodically
        if metrics.should_report():
            metrics.print_stats(detections_count=2)

    print("\n✓ Metrics test complete")
