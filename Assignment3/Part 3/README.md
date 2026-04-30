# Part C: Pose Estimation (Yoga Poses)

## 🎯 Objective
To train and evaluate a YOLO pose estimation model to detect human keypoints and classify yoga poses.

## 📊 Dataset Information
- **Dataset**: Yoga Pose - v2
- **Source**: [Roboflow Universe - Yoga Pose](https://universe.roboflow.com/yoga-pose-prediction-riset-abdimas-2024/yoga-pose-uq4bq/dataset/2)
- **Images**: 724
- **Keypoints**: 14 (Skeleton format)
- **Classes**: 5 (Downdog, Goddess, Plank, Tree, Warrior2)

## 🛠️ Implementation Details
- **Model**: YOLOv11 Nano Pose (`yolo11n-pose.pt`)
- **Epochs**: 10
- **Input Size**: 640x640

## 📂 Files in this Folder
- `train_pose.py`: Script to train the pose model on the yoga dataset.
- `test_pose.py`: Script to run inference on test images to visualize keypoints.
- `yoga_pose_best.pt`: (Generated after training) The best model weights.

## 🚀 How to Run
1. **Training**:
   ```bash
   python train_pose.py
   ```
2. **Testing**:
   ```bash
   python test_pose.py
   ```
