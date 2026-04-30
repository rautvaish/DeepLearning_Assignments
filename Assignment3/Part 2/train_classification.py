from ultralytics import YOLO
import torch
import os

def main():
    print("🚀 Initializing YOLOv11 Nano for Image Classification Training 🚀")
    
    # Check System Configuration
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define paths
    # The classification folder contains train/test/valid folders
    dataset_path = r"c:\Users\prana\OneDrive\Desktop\dl practicle 3\classification"
    
    # 1. Load a pretrained YOLOv11 classification model
    print("\n--- 1. Loading Classification Model ---")
    model = YOLO("yolo11n-cls.pt")
    
    # 2. Train the model
    print("\n--- 2. Starting Training (20 Epochs) ---")
    results = model.train(
        data=dataset_path, 
        epochs=20,         # As requested by the user
        imgsz=224,         # Standard classification size
        device=device,
        project="Car_Color_Results", 
        name="yolo11n_cls_training"
    )
    
    print("\n✅ Training Complete. Output saved to 'Car_Color_Results/yolo11n_cls_training'")
    
    # 3. Validation
    print("\n--- 3. Running Validation ---")
    metrics = model.val()
    
    # 4. Save the best model locally in the part folder
    best_model_path = os.path.join("Car_Color_Results", "yolo11n_cls_training", "weights", "best.pt")
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, "car_color_best.pt")
        print("✅ Best model weight saved as 'car_color_best.pt'")

if __name__ == "__main__":
    main()
