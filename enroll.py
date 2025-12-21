import sys
import cv2
from ultralytics import YOLO
from arcface_infer import ArcFaceONNX
from crypto_store import save_encrypted_embedding

def main():
    if len(sys.argv) != 3:
        print("Usage: python enroll.py <person_name> <image_path>")
        sys.exit(1)

    person_name = sys.argv[1]
    image_path = sys.argv[2]

    img = cv2.imread(image_path)
    if img is None:
        print("Failed to read image:", image_path)
        sys.exit(1)

    model = YOLO("yolo11n.pt")
    results = model(img)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        print("No detections in enrollment image.")
        sys.exit(1)

    # Use first detection (you can refine later to select class=person/face)
    x1, y1, x2, y2 = boxes[0].astype(int)
    face_crop = img[y1:y2, x1:x2]
    if face_crop.size == 0:
        print("Empty crop, aborting.")
        sys.exit(1)

    arc = ArcFaceONNX("arcface.onnx")
    emb = arc(face_crop)
    save_encrypted_embedding(person_name, emb)
    print(f"Enrolled {person_name} with encrypted embedding.")

if __name__ == "__main__":
    main()
