# Real-Time Human Classification: From Custom Architecture to SOTA

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20Optimized-EE4C2C)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow)

An end-to-end ML project for real-time human detection from live video feeds. This project documents the progression from a rudimentary baseline CNN to a custom-engineered Darknet-19 architecture built from scratch, ultimately benchmarking against the State-of-the-Art (SOTA) YOLOv8 model. 

Trained entirely on Apple Silicon (M2) using PyTorch's Metal Performance Shaders (`mps`).

## Demonstration
<video src="assets/demo.mp4" autoplay loop muted playsinline width="100%"></video>

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