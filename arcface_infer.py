import cv2
import numpy as np
import onnxruntime as ort

class ArcFaceONNX:
    def __init__(self, onnx_path="arcface.onnx"):
        # Use CPU provider for now; CUDA isn't available in this container
        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = (112, 112)

    def preprocess(self, face_bgr: np.ndarray) -> np.ndarray:
        # BGR -> RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        # Resize to 112x112
        face_resized = cv2.resize(face_rgb, self.input_size)
        # ArcFace-style normalization: (img - 127.5) / 128
        face_norm = (face_resized.astype(np.float32) - 127.5) / 128.0
        # Keep NHWC, add batch dim: (1, 112, 112, 3)
        face_nhwc = face_norm[None, ...]
        return face_nhwc

    def __call__(self, face_bgr: np.ndarray) -> np.ndarray:
        inp = self.preprocess(face_bgr)  # (1, 112, 112, 3)
        outputs = self.session.run(None, {self.input_name: inp})
        emb = outputs[0][0]  # (512,)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        return emb.astype(np.float32)
