import cv2
import torch
from ultralytics import YOLO
import time
import os


# 1. CONFIGURATION
# Put the correct path of the video you want to process.

INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output.mp4"

# "yolov8x.pt" is the Extra Large model (Most Accurate / Slower)
# Use "yolov8m.pt" if you want it to run faster.
MODEL_NAME = "model/yolov8m.pt" 

CONF_THRESHOLD = 0.5
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

print(f"Model: {MODEL_NAME} | Device: {DEVICE}")

# 2. LOAD MODEL

model = YOLO(MODEL_NAME)
model.to(DEVICE)

# 3. VIDEO PROCESSING
if not os.path.exists(INPUT_VIDEO):
    print(f"Input video not found.")
    exit()

cap = cv2.VideoCapture(INPUT_VIDEO)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))


frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # 1. Inference
    # classes=0 forces it to ONLY look for persons. 
    # This speeds it up and removes cars/dogs automatically.
    results = model(frame, conf=CONF_THRESHOLD, classes=0, verbose=False)

    # 2. Draw Boxes
    # YOLOv8 returns boxes in [x1, y1, x2, y2] format
    boxes = results[0].boxes
    
    for box in boxes:
        # Get coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        
        # Label
        label = f"Person {int(conf*100)}%"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out.write(frame)
    
    frame_count += 1
    if frame_count % 10 == 0:
        elapsed = time.time() - start_time
        fps_proc = frame_count / elapsed
        print(f"\rProgress: {frame_count}/{total_frames} | Speed: {fps_proc:.1f} FPS", end="")

cap.release()
out.release()
print(f"\n Done'")