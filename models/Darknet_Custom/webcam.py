import torch
import torch.nn as nn
import cv2
import numpy as np
import time
from torchvision import transforms


# 1. Args

MODEL_PATH = "checkpoints_darknet/best_model.pth"

PAD_RATIO = 0.5

ALPHA_POS  = 0.6
ALPHA_SIZE = 0.4

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
        self.tracks = []

    def update(self, new_boxes):
        updated_tracks = []

        for box in new_boxes:
            nx1, ny1, nx2, ny2, conf = box
            nw, nh = nx2 - nx1, ny2 - ny1
            ncx, ncy = nx1 + nw/2, ny1 + nh/2

            matched = False
            for i, track in enumerate(self.tracks):
                tx1, ty1, tx2, ty2 = track
                tw, th = tx2 - tx1, ty2 - ty1
                tcx, tcy = tx1 + tw/2, ty1 + th/2

                dist = np.sqrt((ncx - tcx)**2 + (ncy - tcy)**2)

                if dist < 100:
                    final_cx = ALPHA_POS * ncx + (1 - ALPHA_POS) * tcx
                    final_cy = ALPHA_POS * ncy + (1 - ALPHA_POS) * tcy

                    final_w = ALPHA_SIZE * nw + (1 - ALPHA_SIZE) * tw
                    final_h = ALPHA_SIZE * nh + (1 - ALPHA_SIZE) * th

                    fx1 = final_cx - final_w / 2
                    fy1 = final_cy - final_h / 2
                    fx2 = final_cx + final_w / 2
                    fy2 = final_cy + final_h / 2

                    updated_tracks.append([fx1, fy1, fx2, fy2])
                    self.tracks.pop(i)
                    matched = True
                    break

            if not matched:
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

    def forward(self, x):
        return self.block(x)


class Darknet19(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32), nn.MaxPool2d(2, 2),
            ConvBlock(32, 64), nn.MaxPool2d(2, 2),
            ConvBlock(64, 128), ConvBlock(128, 64, k=1, p=0),
            ConvBlock(64, 128), nn.MaxPool2d(2, 2),
            ConvBlock(128, 256), ConvBlock(256, 128, k=1, p=0),
            ConvBlock(128, 256), nn.MaxPool2d(2, 2),
            ConvBlock(256, 512), ConvBlock(512, 256, k=1, p=0),
            ConvBlock(256, 512), ConvBlock(512, 256, k=1, p=0),
            ConvBlock(256, 512), nn.MaxPool2d(2, 2),
            ConvBlock(512, 1024), ConvBlock(1024, 512, k=1, p=0),
            ConvBlock(512, 1024), ConvBlock(1024, 512, k=1, p=0),
            ConvBlock(512, 1024)
        )
        self.head = nn.Conv2d(1024, 30, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x.permute(0, 2, 3, 1).contiguous().view(x.size(0), 13, 13, 5, 6)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def decode_prediction(output, anchors):
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

                    x1 = (cx - w/2) * IMG_SIZE
                    y1 = (cy - h/2) * IMG_SIZE
                    x2 = (cx + w/2) * IMG_SIZE
                    y2 = (cy + h/2) * IMG_SIZE

                    boxes.append([x1, y1, x2, y2, conf])

    return boxes


def nms(boxes):
    if not boxes:
        return []

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

            xx1 = max(cx1, bx1)
            yy1 = max(cy1, by1)
            xx2 = min(cx2, bx2)
            yy2 = min(cy2, by2)

            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)

            inter = w * h
            area_b = (bx2 - bx1) * (by2 - by1)
            iou = inter / (area_c + area_b - inter)

            if iou < NMS_THRESHOLD:
                new_boxes.append(box)

        boxes = new_boxes

    return keep


# 4. LIVE WEBCAM EXECUTION

if __name__ == "__main__":

    model = Darknet19().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    stabilizer = BoxStabilizer()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam")
        exit()

    print("Press 'Q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]

        pad_w = int(width * PAD_RATIO)
        pad_h = int(height * PAD_RATIO)

        padded_w = width + 2 * pad_w
        padded_h = height + 2 * pad_h

        canvas = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)
        canvas[pad_h:pad_h+height, pad_w:pad_w+width] = frame

        img_resized = cv2.resize(canvas, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        input_tensor = transforms.ToTensor()(img_rgb).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(input_tensor)

        boxes = decode_prediction(output, ANCHORS)
        raw_boxes = nms(boxes)

        # Rescale back
        scale_x = padded_w / IMG_SIZE
        scale_y = padded_h / IMG_SIZE

        norm_boxes = []
        for box in raw_boxes:
            x1, y1, x2, y2, conf = box
            norm_boxes.append([
                x1 * scale_x,
                y1 * scale_y,
                x2 * scale_x,
                y2 * scale_y,
                conf
            ])

        smoothed_tracks = stabilizer.update(norm_boxes)

        for (sx1, sy1, sx2, sy2) in smoothed_tracks:
            final_x1 = max(0, sx1 - pad_w)
            final_y1 = max(0, sy1 - pad_h)
            final_x2 = min(width, sx2 - pad_w)
            final_y2 = min(height, sy2 - pad_h)

            if final_x2 > final_x1 and final_y2 > final_y1:
                cv2.rectangle(
                    frame,
                    (int(final_x1), int(final_y1)),
                    (int(final_x2), int(final_y2)),
                    (0, 255, 0),
                    4
                )

        cv2.imshow("Live Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")
