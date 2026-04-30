from ultralytics import YOLO
import torch
import os

def main():
    print("🚀 Initializing YOLOv11 Nano for Pose Estimation Training 🚀")
    
    # Check System Configuration
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define paths
    dataset_yaml = r"c:\Users\prana\OneDrive\Desktop\dl practicle 3\Pose Estimation\Yoga Pose.v2i.yolov8\data.yaml"
    
    # 1. Load a pretrained YOLOv11 pose model
    print("\n--- 1. Loading Pose Model ---")
    model = YOLO("yolo11n-pose.pt")
    
    # 2. Train the model
    print("\n--- 2. Starting Training ---")
    results = model.train(
        data=dataset_yaml, 
        epochs=10,         # Number of epochs
        imgsz=640,        # Standard YOLO size
        device=device,
        project="Yoga_Pose_Results", 
        name="yolo11n_pose_training"
    )
    
    print("\n✅ Training Complete. Output saved to 'Yoga_Pose_Results/yolo11n_pose_training'")
    
    # 3. Save the best model locally in the part folder
    best_model_path = os.path.join("Yoga_Pose_Results", "yolo11n_pose_training", "weights", "best.pt")
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, "yoga_pose_best.pt")
        print("✅ Best model weight saved as 'yoga_pose_best.pt'")

if __name__ == "__main__":
    main()
