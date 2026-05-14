import cv2
import torch
from ultralytics import YOLO
import time

# CONFIG 

MODEL_NAME = "model/yolov8m.pt"   # Put the model path
CONF_THRESHOLD = 0.55
IMG_SIZE = 768 
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Model :{MODEL_NAME} | Device: {DEVICE}")

# Load model

model = YOLO(MODEL_NAME)
model.to(DEVICE)



cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Webcam started. Press 'Q' to quit")

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference (Tracking mode for stability)
    results = model.track(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        classes=0,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            track_id = int(box.id[0]) if box.id is not None else -1

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            label = f"ID {track_id} | {int(conf*100)}%"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # FPS Counter
    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.imshow("YOLOv8 Live - Balanced Accuracy", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
