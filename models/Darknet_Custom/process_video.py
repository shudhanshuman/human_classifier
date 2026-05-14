import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import time
from torchvision import transforms


# 1. CONFIGURATION

INPUT_VIDEO = "input_video.mp4"  # Change this to your input video path
OUTPUT_VIDEO = "output_video.mp4" # Output path
MODEL_PATH = "checkpoints_darknet/best_model.pth"

# PADDING (Data set bias fix)
PAD_RATIO = 0.5

# SMOOTHING FACTORS 
# alpha close to 1.0 = Trust the Model
# alpha close to 0.0 = Trust past tracks
ALPHA_POS  = 0.6  
ALPHA_SIZE = 0.4 

# Model Settings
IMG_SIZE = 416
CONF_THRESHOLD = 0.7
NMS_THRESHOLD = 0.3
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else "cpu"

ANCHORS = [
    (1.32, 1.73), (3.19, 4.00), (5.05, 8.09), 
    (9.47, 4.84), (11.23, 10.00)
]

print(f"Device: {DEVICE}")

# 2. STABILIZER

class BoxStabilizer:
    def __init__(self):
        # Stores the state of known boxes: {id: [x1, y1, x2, y2]}
        self.tracks = [] 
        
    def update(self, new_boxes):
        # new_boxes format: [[x1, y1, x2, y2, conf]]
        
        updated_tracks = []
        
        # Simple centroid matching
        for box in new_boxes:
            nx1, ny1, nx2, ny2, conf = box
            nw, nh = nx2 - nx1, ny2 - ny1
            ncx, ncy = nx1 + nw/2, ny1 + nh/2
            
            matched = False
            for i, track in enumerate(self.tracks):
                tx1, ty1, tx2, ty2 = track
                tw, th = tx2 - tx1, ty2 - ty1
                tcx, tcy = tx1 + tw/2, ty1 + th/2
                
                # If centroids are close enough, consider it the same person
                dist = np.sqrt((ncx - tcx)**2 + (ncy - tcy)**2)
                
                # Threshold: If centers are within 100 pixels, it's the same person
                if dist < 100: 
                    
                    # 1. liner Smoothing 
                    final_cx = ALPHA_POS * ncx + (1 - ALPHA_POS) * tcx
                    final_cy = ALPHA_POS * ncy + (1 - ALPHA_POS) * tcy
                    
                    # 2. size Smoothing
                    final_w = ALPHA_SIZE * nw + (1 - ALPHA_SIZE) * tw
                    final_h = ALPHA_SIZE * nh + (1 - ALPHA_SIZE) * th
                    
                    # Reconstruct box
                    fx1 = final_cx - final_w / 2
                    fy1 = final_cy - final_h / 2
                    fx2 = final_cx + final_w / 2
                    fy2 = final_cy + final_h / 2
                    
                    updated_tracks.append([fx1, fy1, fx2, fy2])
                    
                    # Remove used track
                    self.tracks.pop(i)
                    matched = True
                    break
            
            if not matched:
                # New person found, start tracking immediately
                updated_tracks.append([nx1, ny1, nx2, ny2])
        
        self.tracks = updated_tracks
        return self.tracks


# 3. MODEL

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x): return self.block(x)

class Darknet19(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),    nn.MaxPool2d(2, 2),
            ConvBlock(32, 64),   nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),  ConvBlock(128, 64, k=1, p=0), ConvBlock(64, 128), nn.MaxPool2d(2, 2),
            ConvBlock(128, 256), ConvBlock(256, 128, k=1, p=0), ConvBlock(128, 256), nn.MaxPool2d(2, 2),
            ConvBlock(256, 512), ConvBlock(512, 256, k=1, p=0), ConvBlock(256, 512), 
            ConvBlock(512, 256, k=1, p=0), ConvBlock(256, 512), nn.MaxPool2d(2, 2),
            ConvBlock(512, 1024), ConvBlock(1024, 512, k=1, p=0), ConvBlock(512, 1024),
            ConvBlock(1024, 512, k=1, p=0), ConvBlock(512, 1024)
        )
        self.head = nn.Conv2d(1024, 30, kernel_size=1)
    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x.permute(0, 2, 3, 1).contiguous().view(x.size(0), 13, 13, 5, 6)

def sigmoid(x): return 1 / (1 + np.exp(-x))

def decode_prediction(output, anchors, img_size):
    boxes = []
    output = output.cpu().detach().numpy()[0]
    S = 13
    for i in range(S):
        for j in range(S):
            for k in range(5):
                pred = output[i, j, k]
                conf = sigmoid(pred[0])
                if conf > CONF_THRESHOLD:
                    x_off, y_off = pred[1], pred[2]
                    w_log, h_log = pred[3], pred[4]
                    cx = (j + x_off) / S
                    cy = (i + y_off) / S
                    w = (np.exp(w_log) * anchors[k][0]) / S
                    h = (np.exp(h_log) * anchors[k][1]) / S
                    x1 = (cx - w/2) * img_size
                    y1 = (cy - h/2) * img_size
                    x2 = (cx + w/2) * img_size
                    y2 = (cy + h/2) * img_size
                    boxes.append([x1, y1, x2, y2, conf])
    return boxes

def nms(boxes, iou_thresh):
    if not boxes: return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    keep = []
    while boxes:
        chosen = boxes.pop(0)
        keep.append(chosen)
        cx1, cy1, cx2, cy2, _ = chosen
        area_c = (cx2 - cx1) * (cy2 - cy1)
        new_boxes = []
        for box in boxes:
            bx1, by1, bx2, by2, _ = box
            xx1 = max(cx1, bx1); yy1 = max(cy1, by1)
            xx2 = min(cx2, bx2); yy2 = min(cy2, by2)
            w = max(0, xx2 - xx1); h = max(0, yy2 - yy1)
            inter = w * h
            area_b = (bx2 - bx1) * (by2 - by1)
            iou = inter / (area_c + area_b - inter)
            if iou < iou_thresh: new_boxes.append(box)
        boxes = new_boxes
    return keep


# 4. MAIN LOOP

if __name__ == "__main__":

    print(f"Loading Model from {MODEL_PATH}...")
    model = Darknet19().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Initialize Stabilizer
    stabilizer = BoxStabilizer()

    cap = cv2.VideoCapture(INPUT_VIDEO)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pad_w = int(width * PAD_RATIO)
    pad_h = int(height * PAD_RATIO)
    padded_w = width + 2 * pad_w
    padded_h = height + 2 * pad_h

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # 1. PADDING
        canvas = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
        canvas[pad_h:pad_h+height, pad_w:pad_w+width] = frame

        # 2. INFERENCE
        img_resized = cv2.resize(canvas, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        input_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)

        boxes = decode_prediction(output, ANCHORS, IMG_SIZE)
        raw_boxes = nms(boxes, NMS_THRESHOLD)

        # 3. TRANSFORM COORDS
        # We need to process these boxes before stabilizing
        norm_boxes = []
        scale_x = padded_w / IMG_SIZE
        scale_y = padded_h / IMG_SIZE

        for box in raw_boxes:
            x1, y1, x2, y2, conf = box
            real_x1 = x1 * scale_x
            real_y1 = y1 * scale_y
            real_x2 = x2 * scale_x
            real_y2 = y2 * scale_y
            norm_boxes.append([real_x1, real_y1, real_x2, real_y2, conf])

        # 4. STABILIZE
        smoothed_tracks = stabilizer.update(norm_boxes)

        # 5. DRAW
        for (sx1, sy1, sx2, sy2) in smoothed_tracks:
            # Remove Padding Offset
            final_x1 = sx1 - pad_w
            final_y1 = sy1 - pad_h
            final_x2 = sx2 - pad_w
            final_y2 = sy2 - pad_h

            # Clip
            final_x1 = max(0, final_x1)
            final_y1 = max(0, final_y1)
            final_x2 = min(width, final_x2)
            final_y2 = min(height, final_y2)

            if final_x2 > final_x1 and final_y2 > final_y1:
                cv2.rectangle(frame, (int(final_x1), int(final_y1)), (int(final_x2), int(final_y2)), (0, 255, 0), 10)

        out.write(frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"\rProcessing {frame_count}/{total_frames}...", end="")

    cap.release()
    out.release()
    print(f"\n Video saved to '{OUTPUT_VIDEO}'")