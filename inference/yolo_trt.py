#!/usr/bin/env python3
"""
TensorRT YOLO Inference Engine

Loads and runs YOLO object detection using TensorRT for GPU acceleration.
"""

import time
import numpy as np
import cv2
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("WARNING: TensorRT/PyCUDA not available, will use ONNX Runtime fallback")
    import onnxruntime as ort


class YOLOTensorRT:
    """TensorRT-accelerated YOLO inference engine"""

    def __init__(self, engine_path, conf_threshold=0.5, iou_threshold=0.4):
        """
        Initialize TensorRT YOLO engine

        Args:
            engine_path: Path to TensorRT engine file
            conf_threshold: Confidence threshold for detections
            iou_threshold: IOU threshold for NMS
        """
        self.engine_path = Path(engine_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.input_size = 640

        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

        if TRT_AVAILABLE and self.engine_path.suffix == '.trt':
            self._load_tensorrt_engine()
        else:
            self._load_onnx_fallback()

    def _load_tensorrt_engine(self):
        """Load TensorRT engine"""
        print(f"Loading TensorRT engine: {self.engine_path}")

        # Create logger
        self.logger = trt.Logger(trt.Logger.WARNING)

        # Load engine
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate buffers
        self.inputs, self.outputs, self.bindings = self._allocate_buffers()
        self.stream = cuda.Stream()

        print(f"✓ TensorRT engine loaded successfully")
        self.backend = "tensorrt"

    def _allocate_buffers(self):
        """Allocate CUDA buffers for input/output"""
        inputs, outputs, bindings = [], [], []

        for i in range(self.engine.num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))

            # Calculate size
            size = trt.volume(shape)

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})

        return inputs, outputs, bindings

    def _load_onnx_fallback(self):
        """Fallback to ONNX Runtime if TensorRT unavailable"""
        onnx_path = self.engine_path.parent / (self.engine_path.stem + '.onnx')

        if not onnx_path.exists():
            raise FileNotFoundError(f"Neither TensorRT engine nor ONNX model found at {onnx_path}")

        print(f"Loading ONNX model: {onnx_path}")

        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)

        print(f"✓ ONNX model loaded (provider: {self.session.get_providers()[0]})")
        self.backend = "onnx"

    def preprocess(self, image):
        """
        Preprocess image for YOLO

        Args:
            image: Input image (BGR format)

        Returns:
            Preprocessed tensor ready for inference
        """
        # Resize to input size
        resized = cv2.resize(image, (self.input_size, self.input_size))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # HWC -> CHW
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(transposed, 0)

        return batched.astype(np.float32)

    def infer_tensorrt(self, input_tensor):
        """Run TensorRT inference"""
        # Copy input to device
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )

        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # Copy output back
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )
        self.stream.synchronize()

        # Reshape output
        output_shape = self.outputs[0]['shape']
        output = self.outputs[0]['host'].reshape(output_shape)

        return output

    def infer_onnx(self, input_tensor):
        """Run ONNX Runtime inference"""
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        return outputs[0]

    def postprocess(self, output, orig_shape):
        """
        Postprocess YOLO output to extract detections

        Args:
            output: Raw model output
            orig_shape: Original image shape (H, W)

        Returns:
            List of detections with format:
            [{'class_id', 'class_name', 'confidence', 'bbox': {'x1', 'y1', 'x2', 'y2'}}]
        """
        detections = []

        # YOLOv8 output format: (1, 84, 8400) - transpose to (8400, 84)
        if len(output.shape) == 3:
            output = output[0].transpose()  # (8400, 84)

        # Each detection: [x, y, w, h, class_scores...]
        boxes = output[:, :4]
        scores = output[:, 4:]

        # Get class with highest score for each detection
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter by confidence
        mask = confidences > self.conf_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes) == 0:
            return []

        # Convert from center format to corner format
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        # Scale to original image size
        orig_h, orig_w = orig_shape
        scale_x = orig_w / self.input_size
        scale_y = orig_h / self.input_size

        x1 = (x1 * scale_x).astype(int)
        y1 = (y1 * scale_y).astype(int)
        x2 = (x2 * scale_x).astype(int)
        y2 = (y2 * scale_y).astype(int)

        # Apply NMS
        indices = self._nms(boxes, confidences)

        for idx in indices:
            detections.append({
                'class_id': int(class_ids[idx]),
                'class_name': self.class_names[class_ids[idx]],
                'confidence': float(confidences[idx]),
                'bbox': {
                    'x1': int(x1[idx]),
                    'y1': int(y1[idx]),
                    'x2': int(x2[idx]),
                    'y2': int(y2[idx])
                }
            })

        return detections

    def _nms(self, boxes, scores):
        """Non-maximum suppression"""
        # Convert to x1, y1, x2, y2 format for NMS
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Compute IoU
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= self.iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def __call__(self, image):
        """
        Run inference on image

        Args:
            image: Input image (BGR format)

        Returns:
            detections: List of detected objects
            metrics: Dict with timing information
        """
        orig_shape = image.shape[:2]

        # Preprocessing
        t0 = time.perf_counter()
        input_tensor = self.preprocess(image)
        t1 = time.perf_counter()

        # Inference
        if self.backend == "tensorrt":
            output = self.infer_tensorrt(input_tensor)
        else:
            output = self.infer_onnx(input_tensor)
        t2 = time.perf_counter()

        # Postprocessing
        detections = self.postprocess(output, orig_shape)
        t3 = time.perf_counter()

        metrics = {
            'preprocessing_ms': (t1 - t0) * 1000,
            'inference_ms': (t2 - t1) * 1000,
            'postprocessing_ms': (t3 - t2) * 1000,
            'total_ms': (t3 - t0) * 1000
        }

        return detections, metrics


if __name__ == "__main__":
    # Test the inference engine
    engine_path = Path(__file__).parent.parent / "models" / "yolov8n.trt"

    if not engine_path.exists():
        print(f"Error: Engine not found at {engine_path}")
        print("Run: python3 scripts/convert_to_trt.py")
        sys.exit(1)

    print("Initializing YOLO TensorRT engine...")
    yolo = YOLOTensorRT(str(engine_path))

    # Test with dummy image
    dummy_image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    print("\nRunning test inference...")
    detections, metrics = yolo(dummy_image)

    print(f"\nResults:")
    print(f"  Detections: {len(detections)}")
    print(f"  Preprocessing: {metrics['preprocessing_ms']:.2f} ms")
    print(f"  Inference: {metrics['inference_ms']:.2f} ms")
    print(f"  Postprocessing: {metrics['postprocessing_ms']:.2f} ms")
    print(f"  Total: {metrics['total_ms']:.2f} ms")
    print(f"\n✓ Inference engine test successful!")
