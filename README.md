# 😷 Real-Time Face Mask Detection using YOLOv8

An end-to-end computer vision project for **real-time face mask compliance detection** using the **YOLOv8 object detection framework**.
The system accurately detects faces and classifies them into three categories based on mask usage.

---

## 🎯 Project Objective

The objective of this project is to automate **safety monitoring in public places** such as hospitals, airports, offices, and schools by detecting whether individuals are:

* Wearing a face mask correctly
* Not wearing a face mask
* Wearing a face mask incorrectly (not covering nose or mouth)

The model is optimized for **real-time performance** and **high detection accuracy**.

---

## 🧠 Classes Detected

The YOLOv8 model is trained on **three classes**:

| Class ID | Class Name            |
| -------- | --------------------- |
| 0        | With Mask             |
| 1        | Without Mask          |
| 2        | Mask Worn Incorrectly |

---

## 🛠️ Tech Stack

* **Model Architecture**: YOLOv8 (Ultralytics)
* **Deep Learning Framework**: PyTorch
* **Computer Vision**: OpenCV
* **Data Handling**: NumPy, Pandas
* **Annotation Format**: YOLO (Normalized Center-Based)
* **Training Environment**: Python (Anaconda / venv)

---

## 📊 Evaluation & Results

### 🔹 Confusion Matrix

The confusion matrix illustrates the classification performance of the model on the validation dataset.
High values along the diagonal indicate strong true-positive detection for each class.


![Confusion Matrix](docs/confusion_matrix.png)


---

### 🔹 Model Predictions vs Ground Truth

Below is a comparison between **human-annotated labels** and **model predictions**, showing accurate localization and classification even in complex scenes.

|         Ground Truth (Labels)         |             Model Predictions            |
| :-----------------------------------: | :--------------------------------------: |
| ![Labels](docs/val_batch0_labels.jpg) | ![Predictions](docs/val_batch0_pred.jpg) |

---

## 📁 Dataset Preparation

* Original annotations were in **PASCAL VOC (XML)** format
* A **custom XML-to-YOLO conversion script** was written
* Bounding boxes were converted to:

  ```
  class_id, x_center, y_center, width, height
  ```
* All values were normalized between **0 and 1**

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/AlsoMeParth/Face-Mask-Detection-yolov8.git
cd face-mask-detection-yolov8
```

---

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 3️⃣ Train the YOLOv8 Model

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640
)
```

Training outputs (weights, metrics, plots) are saved in:

```
runs/detect/train/
```

---

### 4️⃣ Live Webcam Inference

To run real-time face mask detection using your webcam:

```bash
python scripts/detect_live.py
```

---

## 📈 Training Logs & Metrics

* YOLOv8 logs training metrics to:

  * `results.csv`
  * `results.png`
  * Precision–Recall & F1 curves
* TensorBoard is **not enabled by default** in Ultralytics YOLO
* Metrics visualization is handled internally by YOLO

---

## 📂 Repository Structure

```
face-mask-detection-yolov8/
│
├── models/
│   └── best.pt
│
├── scripts/
│   ├── xml_to_yolo.py
│   └── detect_live.py
│
├── docs/
│   ├── confusion_matrix.png
│   ├── val_batch0_labels.jpg
│   └── val_batch0_pred.jpg
│
├── runs/
│   └── detect/train/
│
├── data.yaml
├── requirements.txt
└── README.md
```

---


## 📜 Conclusion

This project demonstrates the effectiveness of **YOLOv8 for real-time face mask detection**, making it suitable for practical deployment in real-world safety and surveillance applications.

---


