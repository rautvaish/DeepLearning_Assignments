# Part A: Object Detection (Face Detection)

## 🎯 Objective
To train and deploy a modern YOLO model for real-time face detection using a custom dataset from Roboflow.

## 📊 Dataset Information
- **Source**: [Roboflow Universe - Face Detection](https://universe.roboflow.com/mohamed-traore-2ekkp/face-detection-mik1i/dataset/27)
- **Format**: YOLOv11
- **Classes**: 1 (Face)

## 🛠️ Implementation Details
- **Model**: YOLOv11 Nano (`yolo11n.pt`)
- **Framework**: Ultralytics YOLO
- **Training Device**: Local (CPU/GPU)
- **Epochs**: 10 (as configured in `train_detection.py`)

## 📂 Files in this Folder
- `train_detection.py`: Script to load the dataset, train the YOLOv11 model, and run validation.
- `webcam_test.py`: Script to run real-time inference using the trained model on a local webcam.
- `face_detection_best.pt`: The optimized weights (best performing model) generated during training.

## 🚀 How to Run
1. **Training**:
   ```bash
   python train_detection.py
   ```
2. **Webcam Inference**:
   ```bash
   python webcam_test.py
   ```

## 📈 Results
- The model weights `face_detection_best.pt` are used for live testing.
- Real-time detection with a confidence threshold of `0.25`.
