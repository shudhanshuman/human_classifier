# Real-Time Human Classification: From Custom Architecture to SOTA

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20Optimized-EE4C2C)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow)

An end-to-end ML project for real-time human detection from live video feeds. This project documents the progression from a rudimentary baseline CNN to a custom-engineered Darknet-19 architecture built from scratch, ultimately benchmarking against the State-of-the-Art (SOTA) YOLOv8 model. 

Trained entirely on Apple Silicon (M2) using PyTorch's Metal Performance Shaders (`mps`).

## Demonstration

![demo.gif](assets/demo.gif)



## Project Overview
This repository systematically assesses three different approaches for object localization, generalization, and temporal stability:
1. **Model 1: Baseline CNN** - A lightweight (1.4M parameter) model utilizing temporal frame differences.
2. **Model 2: Custom Darknet-19** - A ~21M parameter anchor-based detector built from scratch, trained for 41 hours on a filtered COCO subset. Features custom temporal smoothing and dynamic padding to mitigate scale bias.
3. **Model 3: YOLOv8 (SOTA)** - An out-of-the-box benchmark to evaluate the custom architecture against modern anchor-free performance.

👉 **[Read the Full Technical Documentation](docs/documentation.md)** for an in-depth breakdown of the architecture, loss curves, engineering challenges, and M2 hardware optimization.

## Repository Structure
```text
HUMAN CLASSIFIER/
├── assets/                  
├── data/                    
│   ├── cleanup.py           
│   └── DATA_SET.md          
├── docs/
│   └── documentation.md     
├── models/
│   ├── CNN_Basic/           
│   ├── Darknet_Custom/      
│   └── YOLOv8_SOTA/ 
├── .gitignore
├── requirements.txt      
└── README.md
```
## Quick Start & Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/human-classifier.git
    cd human-classifier
    ```

2. **Install dependencies:**
    
    ```bash
    pip install -r requirements.txt
    ```
3. **Download Model Weights**

     Download `best_model.pth` from the [Custom Model](https://github.com/shudhanshuman/human_classifier/releases/tag/v1.0) and place it inside `models/Darknet_Custom/checkpoints_darknet/`
4. **Dataset Preparation (Only if you want to train your own model):**

   Because the raw dataset is ~18GB, it is not hosted in this repository. Please follow the instructions in `data/DATA_SET.md` to download the COCO 2017 subset and run the `cleanup.py` script.

##  Running the Models (Live Webcam)

You can test each model's real-time inference on your local webcam. Navigate to the respective model directory and run the webcam script.

**Run the Custom Darknet Model:**
```bash
cd models/Darknet_Custom
python webcam.py
```

**Run the YOLOv8 Benchmark:**
```bash
cd models/YOLOv8_SOTA
python webcam.py
```

##  Performance Matrix

| Feature | Baseline CNN | Custom Darknet | YOLOv8 (SOTA) |
| :--- | :--- | :--- | :--- |
| **Accuracy (mAP)** | Very low | 68% | 94% |
| **Model Size** | 4.6 MB | ~80 MB | 87 MB |
| **Real-Time Inference** | Yes | Yes | Yes |
| **Architecture Type** | Frame-difference | Anchor-based | Anchor-free |

##  Key Learnings

* **Data Engineering:** Curating and pruning a large data set like COCO.
* **Training Stability:** Executed a 41-hour stable training run while maintaining high Model FLOPs Utilization (MFU), optimizing the pipeline to ensure hardware efficiency and consistent gradient flow without compute bottlenecks.  
* **Inference Pipeline:** Achieving a stable video output required engineering dynamic padding (to fix proximity scale bias) and a custom temporal smoothing algorithm (`process_video.py`) to eliminate bounding box jitter.

---
*Developed by Shudhanshu Ranjan Gupta, ECE , IIT Guwahati*
