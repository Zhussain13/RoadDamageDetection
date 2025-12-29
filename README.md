🛣️ RoadCare+

### A Lightweight and Efficient Road Surface Damage Detection Framework for Indian Roads

**Using Enhanced YOLOv7 with GhostNet, BasicRFB, and CARAFE**

---

## 📌 Overview

RoadCare+ is a lightweight deep learning framework designed for **real-time road surface damage detection** under Indian road conditions. The system enhances the **YOLOv7** object detection architecture with **GhostNet**, **BasicRFB**, and **CARAFE** modules to achieve a balance between **high accuracy**, **low computational cost**, and **fast inference speed**.

The framework is trained on the **RDD2020 (India subset)** dataset and supports **automated reporting** and **easy deployment**, making it suitable for **smart city infrastructure monitoring** and **municipal road maintenance planning**.

---

## 🎯 Objectives

* Design a **lightweight YOLOv7-based model (GRC-YOLOv7)**
* Improve detection of **potholes and road cracks** with enhanced feature extraction
* Achieve **real-time inference** on CPU/GPU
* Provide **automated Excel & PDF reporting**
* Enable **easy usage** through a deployable inference pipeline

---

## 🧠 Key Features

* ✅ Lightweight architecture using **GhostNet**
* ✅ Improved receptive field via **BasicRFB**
* ✅ High-quality upsampling using **CARAFE**
* ✅ Supports **4 road damage classes**
* ✅ Pre-trained models included
* ✅ Test video inference supported
* ✅ Modular and extendable design

---

## 🧩 Road Damage Classes

| Class ID | Damage Type        |
| -------- | ------------------ |
| D00      | Longitudinal Crack |
| D10      | Transverse Crack   |
| D20      | Alligator Crack    |
| D40      | Pothole            |

---

## 📂 Project Structure

```
RoadCare+
│── Main.py
│── README.md
│
├── Docs
│   ├── Final Copy.pdf
│   ├── Mini Project.pptx
│   ├── RDD2022 A multi-national image dataset.pdf
│   ├── Road Damage Detection Algorithm.pdf
│   ├── Road damage detection using deep neural networks.pdf
│   └── GAN-based road damage detection.pdf
│
├── Test_data
│   └── test_video_1.mp4
│
└── Trained_Models
    ├── classes.names
    ├── pothole_detection_basicyolov7.pt
    └── Pothole_detection_yolov7customised.pt
```

---

## ⚙️ Methodology

1. **Dataset Preparation**

   * RDD2020 India subset
   * YOLO format annotations
   * Image resizing & augmentation

2. **Model Architecture (GRC-YOLOv7)**

   * GhostNet → parameter reduction
   * BasicRFB → contextual feature learning
   * CARAFE → content-aware upsampling

3. **Training & Optimization**

   * PyTorch framework
   * SGD optimizer with cosine LR scheduling
   * Early stopping & mixed precision training

4. **Inference & Evaluation**

   * Bounding box prediction
   * Confidence thresholding
   * NMS post-processing

---

## 📊 Performance Metrics

| Metric     | Value     |
| ---------- | --------- |
| mAP@0.5    | **70.6%** |
| Precision  | 71%       |
| Recall     | 69%       |
| F1-Score   | 0.70      |
| FPS (CPU)  | ~35       |
| FPS (GPU)  | ~62       |
| Model Size | ~12.5 MB  |

---

## 🚀 How to Run

### 1️⃣ Clone the Repository

```bash
git clone git@github.com:Zhussain13/RoadDamageDetection.git
cd RoadDamageDetection
```

### 2️⃣ Install Dependencies

```bash
pip install torch torchvision opencv-python numpy pandas matplotlib
```

### 3️⃣ Run Inference

```bash
python Main.py
```

> Ensure the required `.pt` model file is selected inside `Main.py`.

---

## 📁 Pretrained Models

Located in `Trained_Models/`

* `pothole_detection_basicyolov7.pt`
* `Pothole_detection_yolov7customised.pt`
* `classes.names`

---

## 🎥 Test Data

Sample test video provided in:

```
Test_data/test_video_1.mp4
```

---

## 🏙️ Applications

* Smart city road monitoring
* Municipal road maintenance planning
* Accident prevention systems
* Autonomous driving perception
* Infrastructure inspection automation

---

📜 License

This project is intended for **academic and research purposes**.

