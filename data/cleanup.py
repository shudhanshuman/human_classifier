import os
import shutil
from pycocotools.coco import COCO
from tqdm import tqdm


ANN_FILE = '../annotations/instances_train2017.json'
IMG_DIR = '../train2017' 

def filter_dataset():
    # 1. Check if files exist
    if not os.path.exists(ANN_FILE):
        print(f"Could not find {ANN_FILE}")
        return
    if not os.path.exists(IMG_DIR):
        print(f"Could not find {IMG_DIR}")
        return

    # 2. Load COCO Data
    try:
        coco = COCO(ANN_FILE)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return

    # 3. Identify Humans
    # We ask COCO: "Give me IDs of all images that contain category 1 (Person)"
    catIds = coco.getCatIds(catNms=['person'])
    imgIds = coco.getImgIds(catIds=catIds)
    
    # Create a set
    keep_files = set()
    print(f"{len(imgIds)} images contain humans.")
    
    for img_info in coco.loadImgs(imgIds):
        keep_files.add(img_info['file_name'])

    # 4. The Purge
    # We scan the actual folder and delete anything NOT in the keep list
    all_files = os.listdir(IMG_DIR)
    
    na_img = 0
    
    for f in tqdm(all_files, desc="Cleaning Dataset"):
        if f.endswith(".jpg"):
            if f in keep_files:
                na_img += 1
            else:
                file_path = os.path.join(IMG_DIR, f)
                os.remove(file_path)

    print(f"{na_img} human images")

if __name__ == "__main__":
    filter_dataset()