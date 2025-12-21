import cv2
import numpy as np
from ultralytics import YOLO
from arcface_infer import ArcFaceONNX
from crypto_store import load_all_embeddings

THRESHOLD = 0.1  # tune this based on testing

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def main():
    model = YOLO("yolo11n.pt")
    arc = ArcFaceONNX("arcface.onnx")
    gallery = load_all_embeddings()
    print("Loaded enrolled people:", list(gallery.keys()))

    cap = cv2.VideoCapture(0)  # change to 1 if your working source is 1
    if not cap.isOpened():
        print("Cannot open camera")
        return

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        results = model(frame)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        print(f"Frame {frame_idx}, detections: {len(boxes)}")

        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            emb = arc(face_crop)

            best_name = "unknown"
            best_score = -1.0
            for name, ref_emb in gallery.items():
                score = cosine_similarity(emb, ref_emb)
                if score > best_score:
                    best_score = score
                    best_name = name

            if best_score >= THRESHOLD:
                print(f"  MATCH: {best_name} (score={best_score:.3f})")
            else:
                print(f"  unknown (best={best_name}, score={best_score:.3f})")

        # headless: no imshow, break after some frames if you want
        if frame_idx >= 1000:  # prevent infinite loop on first test
            break

    cap.release()

if __name__ == "__main__":
    main()
