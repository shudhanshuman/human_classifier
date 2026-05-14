# Dataset Preparation: COCO 2017 (Person Subset)

Due to GitHub's file size limitations, the large 18GB training dataset and the associated model checkpoints are not included in this repository. 

To train the custom Darknet model yourself or run the evaluation scripts, you will need to download the official MS COCO 2017 dataset, structure it correctly, and run the included cleanup script to isolate the human classification data.

## 1. Download the Data

This project utilizes the **COCO 2017 Train/Val annotations** and the **2017 Training images**. You can download them directly from the [official COCO website](https://cocodataset.org/#download) or use the following terminal commands:

**Training Images (18GB - Required):**
```bash
wget http://images.cocodataset.org/zips/train2017.zip
```
**Validation Images (1GB - Optional for pure training, required for eval):**

```bash
wget http://images.cocodataset.org/zips/val2017.zip
```

**Annotations (241MB - Required):**

```Bash
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```
## 2. Extract and Structure the Directory

Extract the downloaded .zip files into your project's working directory. Ensure your folder structure matches this layout exactly, as the cleanup and training scripts rely on these specific paths:

```bash
human-classification/
├── train2017/                   # Extracted .jpg training images
├── val2017/                     # Extracted .jpg validation images
├── annotations/
│   ├── instances_train2017.json # Used for mapping bounding boxes
│   └── instances_val2017.json
├── scripts/
│   └── cleanup.py               # The dataset pruning script
└── README.md
```
## 3. Data Preprocessing & Pruning (cleanup.py)
The standard COCO dataset contains 80 different object classes (dogs, cars, stop signs, etc.). Because this project is strictly focused on Human Classification (Class ID: 1), processing the full 18GB dataset is inefficient.

To optimize training speed and reclaim disk space, run the provided dataset pruning script.

Prerequisites
Ensure you have the COCO API installed before running the script:

Bash
pip install pycocotools tqdm
Running the Purge
Execute the cleanup script from your root project directory:

Bash
python scripts/cleanup.py
What This Script Does:
Parses Annotations: It reads instances_train2017.json and asks the COCO API to identify every image ID that contains at least one person.

Creates a Keep List: It maps those IDs to their specific .jpg filenames and stores them in a highly optimized lookup set.

In-Place Deletion: It scans the train2017 directory and permanently deletes any image file that does not exist in the "keep list."

**⚠️ Warning: This script performs in-place deletion. It will permanently remove non-human images from your local train2017 folder. If you plan to use the full COCO dataset for other machine learning projects on your machine, please ensure you have a backup of the original train2017.zip file.**