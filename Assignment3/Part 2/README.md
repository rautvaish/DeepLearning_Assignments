# Part B: Image Classification (Car Colors)

## 🎯 Objective
To train a YOLO classification model to recognize different car colors from a dataset.

## 📊 Dataset Information
- **Dataset**: [Car Colors - v2](https://universe.roboflow.com/tyler-yonjx/car-colors-1smyc/dataset/2)
- **Source**: Roboflow
- **Images**: 601
- **Preprocessing**: Resized to 640x640 (fit within)

## 🛠️ Implementation Details
- **Model**: YOLOv11 Nano Classification (`yolo11n-cls.pt`)
- **Epochs**: 20
- **Batch Size**: Auto
- **Input Size**: 224x224

## 📂 Files in this Folder
- `train_classification.py`: Script to train the model for 20 epochs.
- `test_classification.py`: Script to run inference on a test image.
- `car_color_best.pt`: (Generated after training) The best model weights.

## 🚀 How to Run
1. **Training**:
   ```bash
   python train_classification.py
   ```
2. **Testing**:
   ```bash
   python test_classification.py
   ```
