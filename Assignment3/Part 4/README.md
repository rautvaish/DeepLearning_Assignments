# Part D: Aerial Detection (Palm Trees)

## 🎯 Objective
To train and evaluate a YOLO model to detect Palm trees from aerial imagery.

## 📊 Dataset Information
- **Dataset**: Aerial - v1
- **Classes**: 2 (Palm, Tree)
- **Format**: YOLOv11 Standard Bounding Boxes

## 🛠️ Implementation Details
- **Model**: YOLOv11 Nano (`yolo11n.pt`)
- **Epochs**: 10
- **Input Size**: 640x640

## 📂 Files in this Folder
- `train_obb.py`: Script to train the model on the aerial dataset.
- `test_obb.py`: Script to run inference on test images.
- `aerial_palm_tree_best.pt`: (Generated after training) The best model weights.

## 🚀 How to Run
1. **Training**:
   ```bash
   python train_obb.py
   ```
2. **Testing**:
   ```bash
   python test_obb.py
   ```
