import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from torchvision import transforms
from tqdm import tqdm

#ModelArgs

# Paths
IMG_DIR = "../../data/train2017"
ANN_FILE = "../../data/annotations/instances_train2017.json"
SAVE_DIR = "checkpoints_darknet"
PLOT_FILE = "loss_plot.png"

# Hyperparameters
IMG_SIZE = 416       # Standard YOLO 
LR = 1e-4            # Starting Learning Rate 
NUM_WORKERS = 4      
WEIGHT_DECAY = 5e-4  
BATCH_SIZE = 32 
EPOCHS = 27

# Hardware
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else "cpu"
print(f"System :{DEVICE}")

os.makedirs(SAVE_DIR, exist_ok=True)

# Anchors (Width, Height) scaled to 13x13 grid
# These allow the model to predict "offsets" rather than raw shapes
ANCHORS = [
    (1.32, 1.73), (3.19, 4.00), (5.05, 8.09), 
    (9.47, 4.84), (11.23, 10.00)
]


# 1.DATA loder 

class COCOPersonDataset(Dataset):
    def __init__(self, img_dir, ann_file, img_size=416, anchors=ANCHORS):
        self.coco = COCO(ann_file)
        self.img_dir = img_dir
        self.img_size = img_size
        self.anchors = anchors
        
        # Filter for Person class (ID: 1)
        self.cat_ids = self.coco.getCatIds(catNms=['person'])
        self.img_ids = self.coco.getImgIds(catIds=self.cat_ids)
        
        # Verification: Remove IDs that don't have files (prevents crashes)
        self.valid_ids = []
        # Check first 100 to verify path structure, then assume rest are good for speed
        # (We assume you ran the cleanup script, so this is just a sanity check)
        for i in self.img_ids[:100]:
            fname = self.coco.loadImgs(i)[0]['file_name']
            if os.path.exists(os.path.join(img_dir, fname)):
                self.valid_ids.append(i)
        if len(self.valid_ids) > 0:
            self.valid_ids = self.img_ids 
            print(f"Training on {len(self.valid_ids)} person images.")
        else:
            print("Error: Images not found")

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

            # Build Targets
            S = 13 # 416 / 32
            targets = torch.zeros((5, S, S, 6)) # [Conf, x, y, w, h, class]
            anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id, catIds=self.cat_ids))
            
            for ann in anns:
                x, y, w, h = ann['bbox']
                
                # Normalize to Grid Coordinates
                cx, cy = (x + w/2) / w_orig * S, (y + h/2) / h_orig * S
                gw, gh = (w / w_orig * S), (h / h_orig * S)
                
                if gw * gh == 0: continue
                
                # Find best anchor (IoU)
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
                    targets[best_k, i, j, 0] = 1.0 # Conf
                    targets[best_k, i, j, 1] = cx - j # Offset X
                    targets[best_k, i, j, 2] = cy - i # Offset Y
                    targets[best_k, i, j, 3] = np.log(max(gw, 1e-6)) # Log W
                    targets[best_k, i, j, 4] = np.log(max(gh, 1e-6)) # Log H
                    targets[best_k, i, j, 5] = 1.0 # Class (Person)

            return img_tensor, targets

        except Exception:
            return torch.zeros((3, self.img_size, self.img_size)), torch.zeros((5, 13, 13, 6))

# 2. MODEL: DARKNET-19 (MODIFIED)

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k=3, s=1, p=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c), # CRITICAL FOR STABILITY
            nn.LeakyReLU(0.1, inplace=True)
        )
    def forward(self, x): return self.block(x)

class Darknet19(nn.Module):
    def __init__(self):
        super().__init__()
        # 19 Convolutions, 5 MaxPools (Downsample 32x)
        self.features = nn.Sequential(
            ConvBlock(3, 32),    nn.MaxPool2d(2, 2),
            ConvBlock(32, 64),   nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),  ConvBlock(128, 64, k=1, p=0), ConvBlock(64, 128), nn.MaxPool2d(2, 2),
            ConvBlock(128, 256), ConvBlock(256, 128, k=1, p=0), ConvBlock(128, 256), nn.MaxPool2d(2, 2),
            # Deep Section
            ConvBlock(256, 512), ConvBlock(512, 256, k=1, p=0), ConvBlock(256, 512), 
            ConvBlock(512, 256, k=1, p=0), ConvBlock(256, 512), nn.MaxPool2d(2, 2),
            # Final Section
            ConvBlock(512, 1024), ConvBlock(1024, 512, k=1, p=0), ConvBlock(512, 1024),
            ConvBlock(1024, 512, k=1, p=0), ConvBlock(512, 1024)
        )
        # Detection Head (1x1 Conv)
        # 5 Anchors * (1 Conf + 4 Coords + 1 Class) = 30 channels
        self.head = nn.Conv2d(1024, 30, kernel_size=1)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        # Reshape: [Batch, 5, 13, 13, 6]
        return x.permute(0, 2, 3, 1).contiguous().view(x.size(0), 13, 13, 5, 6).permute(0, 3, 1, 2, 4)

# 3. LOSS FUNCTION

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.bce = nn.BCEWithLogitsLoss(reduction='sum') # Better for prob/conf
        
    def forward(self, pred, target):
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0

        # Objectness Loss (BCE is stable)
        loss_conf_obj = self.bce(pred[..., 0][obj], target[..., 0][obj])
        loss_conf_noobj = self.bce(pred[..., 0][noobj], target[..., 0][noobj])

        # Coordinate Loss (MSE)
        loss_x = self.mse(pred[..., 1][obj], target[..., 1][obj])
        loss_y = self.mse(pred[..., 2][obj], target[..., 2][obj])
        loss_w = self.mse(pred[..., 3][obj], target[..., 3][obj])
        loss_h = self.mse(pred[..., 4][obj], target[..., 4][obj])

        # Class Loss
        loss_cls = self.bce(pred[..., 5][obj], target[..., 5][obj])

        return (5.0 * (loss_x + loss_y + loss_w + loss_h)) + \
               (1.0 * loss_conf_obj) + \
               (0.5 * loss_conf_noobj) + \
               (1.0 * loss_cls)

# 4. TRAINING LOOP
def train():
    # Setup
    dataset = COCOPersonDataset(IMG_DIR, ANN_FILE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, 
                        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    
    model = Darknet19().to(DEVICE)
    resume_path = "checkpoints_darknet/best_model.pth"

    if os.path.exists(resume_path):
        print(f"Training from {resume_path}...")
        model.load_state_dict(torch.load(resume_path, map_location=DEVICE))

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = YoloLoss()
    
    loss_history = []
    best_loss = float('inf')
    
    print(f"Starting Training")
    start_global = time.time()
    
    try:
        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0
            batch_start = time.time()
            
            pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
            for i, (img, target) in enumerate(pbar):
                img, target = img.to(DEVICE), target.to(DEVICE)
                
                optimizer.zero_grad()
                out = model(img)
                loss = criterion(out, target)
                loss.backward()
                
                # Gradient Clipping (Prevents crashes from exploding gradients)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                optimizer.step()
                
                val = loss.item()
                epoch_loss += val
                loss_history.append(val)
                
                # Update progress bar
                pbar.set_postfix(loss=val)

            # End of Epoch Stats
            avg_loss = epoch_loss / len(loader)
            scheduler.step()
            
            print(f"   Done. Avg Loss: {avg_loss:.4f} | Best: {best_loss:.4f}")
            
            # Save Checkpoint (Fail-Safe)
            torch.save(model.state_dict(), f"{SAVE_DIR}/last_checkpoint.pth")
            
            # Save Best Model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")
                print("New Best Model Saved!")

            # Save every 10 epochs
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), f"{SAVE_DIR}/epoch_{epoch+1}.pth")

    except KeyboardInterrupt:
        print("\n Training Interrupted by User!")
    except Exception as e:
        print(f"\n CRITICAL ERROR: {e}")
    finally:
        # EXECUTE FAIL-SAFE SAVE
        print("Saving Emergency Backup")
        torch.save(model.state_dict(), f"{SAVE_DIR}/emergency_backup.pth")
        
        # VISUALISATION
        print(f"Generating Loss Plot")
        plt.figure(figsize=(12, 6))
        plt.plot(loss_history, label='Batch Loss', alpha=0.5)
        
        # Add a smoothed trend line
        if len(loss_history) > 100:
            kernel_size = 100
            kernel = np.ones(kernel_size) / kernel_size
            smoothed = np.convolve(loss_history, kernel, mode='valid')
            plt.plot(np.arange(len(smoothed)) + kernel_size//2, smoothed, 'r', linewidth=2, label='Trend')

        plt.title(f"Darknet-19 Training Loss ({EPOCHS} Epochs)")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(PLOT_FILE)
        plt.show()
        
        total_time = (time.time() - start_global) / 3600
        print(f"DONE. Total Training Time: {total_time:.2f} hours")

if __name__ == "__main__":
    train()