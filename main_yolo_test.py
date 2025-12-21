from ultralytics import YOLO

def main():
    model = YOLO("yolo11n.pt")
    # No show=True; just run and print summary
    results = model.predict(source=0, save=False, stream=True)
    for i, r in enumerate(results):
        print(f"Frame {i}, detections:", r.boxes.xyxy.shape[0])

if __name__ == "__main__":
    main()
