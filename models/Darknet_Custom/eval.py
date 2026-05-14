import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from pycocotools.coco import COCO
from torchvision import transforms
from tqdm import tqdm

# Args

IMG_DIR = "../../data/val2017"
ANN_FILE = "../../data/annotations/instances_val2017.json"
MODEL_PATH = "checkpoints_darknet/best_model17.pth"

IMG_SIZE = 416
BATCH_SIZE = 32
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else "cpu"

# Standard Anchors
ANCHORS = [
    (1.32, 1.73), (3.19, 4.00), (5.05, 8.09), 
    (9.47, 4.84), (11.23, 10.00)
]

print(f" Device: {DEVICE}")
print(f"Dataset:{IMG_DIR}")

# 1. DATASET loder
class COCOPersonDataset(Dataset):
    def __init__(self, img_dir, ann_file, img_size=416, anchors=ANCHORS):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.anchors = anchors
        
        # Filter for Person class (ID: 1)
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        self.img_ids = self.coco.getImgIds(catIds=self.cat_ids)
        
        # Simple verification
        self.valid_ids = []
        for i in self.img_ids:
            fname = self.coco.loadImgs(i)[0]['file_name']
            if os.path.exists(os.path.join(img_dir, fname)):
                self.valid_ids.append(i)
        
        print(f"{len(self.valid_ids)} validation images.")

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        try:
            img_id = self.valid_ids[idx]
            img_info = self.coco.loadImgs(img_id)[0]
            path = os.path.join(self.img_dir, img_info['file_name'])
            
            img = cv2.imread(path)
            if img is None: raise FileNotFoundError
            
            h_orig, w_orig = img.shape[:2]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_size, self.img_size))
            img_tensor = transforms.ToTensor()(img)

            S = 13
            targets = torch.zeros((5, S, S, 6))
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids))
            
            for ann in anns:
                x, y, w, h = ann['bbox']
                cx, cy = (x + w/2) / w_orig * S, (y + h/2) / h_orig * S
                gw, gh = (w / w_orig * S), (h / h_orig * S)
                if gw * gh == 0: continue
                
                best_iou = 0
                best_k = 0
                for k, (aw, ah) in enumerate(self.anchors):
                    inter = min(gw, aw) * min(gh, ah)
                    union = (gw * gh) + (aw * ah) - inter
                    iou = inter / union
                    if iou > best_iou:
                        best_iou = iou
                        best_k = k
                
                i, j = int(cy), int(cx)
                if i < S and j < S:
                    targets[best_k, i, j, 0] = 1.0
                    targets[best_k, i, j, 1] = cx - j
                    targets[best_k, i, j, 2] = cy - i
                    targets[best_k, i, j, 3] = np.log(max(gw, 1e-6))
                    targets[best_k, i, j, 4] = np.log(max(gh, 1e-6))
                    targets[best_k, i, j, 5] = 1.0

            return img_tensor, targets
        except Exception:
            return torch.zeros((3, self.img_size, self.img_size)), torch.zeros((5, 13, 13, 6))

# Model

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
        return x.permute(0, 2, 3, 1).contiguous().view(x.size(0), 13, 13, 5, 6).permute(0, 3, 1, 2, 4)

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum')
        
    def forward(self, pred, target):
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        loss_conf_obj = self.bce(pred[..., 0][obj], target[..., 0][obj])
        loss_conf_noobj = self.bce(pred[..., 0][noobj], target[..., 0][noobj])
        
        loss_x = self.mse(pred[..., 1][obj], target[..., 1][obj])
        loss_y = self.mse(pred[..., 2][obj], target[..., 2][obj])
        loss_w = self.mse(pred[..., 3][obj], target[..., 3][obj])
        loss_h = self.mse(pred[..., 4][obj], target[..., 4][obj])
        
        loss_cls = self.bce(pred[..., 5][obj], target[..., 5][obj])

        return (5.0 * (loss_x + loss_y + loss_w + loss_h)) + \
               (1.0 * loss_conf_obj) + \
               (0.5 * loss_conf_noobj) + \
               (1.0 * loss_cls)

# 3. EVALUATION LOOP
def evaluate():
    # 1. Load Data
    if not os.path.exists(ANN_FILE):
        print(f"Error: Could not find {ANN_FILE}")
        return
        
    dataset = COCOPersonDataset(IMG_DIR, ANN_FILE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, 
                        num_workers=0, pin_memory=False) # Safe settings for Mac
    
    # 2. Load Model
    model = Darknet19().to(DEVICE)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        print("No model found")
        return
        
    criterion = YoloLoss()
    model.eval() # <--- CRITICAL: Freezes Batch Norm & Dropout
    
    total_loss = 0
    num_batches = len(loader)
    
    
    with torch.no_grad():
        for i, (img, target) in enumerate(tqdm(loader)):
            img, target = img.to(DEVICE), target.to(DEVICE)
            
            out = model(img)
            loss = criterion(out, target)
            total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"/n Loss: {avg_loss:.4f}")
    

if __name__ == "__main__":
    evaluate()