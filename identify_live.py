# identify_live.py
import cv2
import numpy as np
from ultralytics import YOLO
from arcface_infer import ArcFaceONNX
from crypto_store import load_all_embeddings

THRESHOLD = 0.3  # Distance threshold (lower = stricter)

def main():
    # Load all enrolled people
    all_embeddings = load_all_embeddings()
    if not all_embeddings:
        print("No enrolled people found!")
        return
    
    print(f"Loaded {len(all_embeddings)} enrolled people: {list(all_embeddings.keys())}")
    
    # Initialize models
    yolo = YOLO("yolo11n.pt")
    arcface = ArcFaceONNX("arcface.onnx")  # Initialize the class
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    print("Starting identification... Press Ctrl+C to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect people/faces with YOLO
            results = yolo(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Class 0 is 'person' in COCO
                    if cls == 0 and conf > 0.5:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        face_crop = frame[y1:y2, x1:x2]
                        
                        if face_crop.size == 0:
                            continue
                        
                        # Get embedding for detected face using the class
                        live_emb = arcface(face_crop)  # Call the class instance
                        
                        # Compare against ALL enrolled people
                        best_match = None
                        best_distance = float('inf')
                        
                        for person_id, stored_emb in all_embeddings.items():
                            distance = 1 - np.dot(live_emb, stored_emb)
                            
                            if distance < best_distance:
                                best_distance = distance
                                best_match = person_id
                        
                        # Check if best match is good enough
                        if best_distance < THRESHOLD:
                            print(f"✓ IDENTIFIED: {best_match} (distance: {best_distance:.3f})")
                        else:
                            print(f"✗ UNKNOWN (closest: {best_match}, distance: {best_distance:.3f})")
    
    except KeyboardInterrupt:
        print("\nStopping identification...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
