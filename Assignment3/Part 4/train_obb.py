from ultralytics import YOLO
import torch
import os

def main():
    print("🚀 Initializing YOLOv11 Nano for Part D: Aerial Detection 🚀")
    print("(Note: This dataset uses standard bounding boxes for Palm/Tree detection)")
    
    # Check System Configuration
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Define paths
    dataset_yaml = r"c:\Users\prana\OneDrive\Desktop\dl practicle 3\bounding box\Aerial.v1i.yolov11\data.yaml"
    
    # 1. Load a pretrained YOLOv11 model
    # We use yolo11n.pt because the dataset provided is standard detection
    print("\n--- 1. Loading Model ---")
    model = YOLO("yolo11n.pt")
    
    # 2. Train the model
    print("\n--- 2. Starting Training ---")
    results = model.train(
        data=dataset_yaml, 
        epochs=10,         # Training for 10 epochs
        imgsz=640,        # Standard YOLO size
        device=device,
        project="Aerial_Detection_Results", 
        name="yolo11n_aerial_training"
    )
    
    print("\n✅ Training Complete. Output saved to 'Aerial_Detection_Results/yolo11n_aerial_training'")
    
    # 3. Save the best model locally in the part folder
    best_model_path = os.path.join("Aerial_Detection_Results", "yolo11n_aerial_training", "weights", "best.pt")
    if os.path.exists(best_model_path):
        import shutil
        shutil.copy(best_model_path, "aerial_palm_tree_best.pt")
        print("✅ Best model weight saved as 'aerial_palm_tree_best.pt'")

if __name__ == "__main__":
    main()
