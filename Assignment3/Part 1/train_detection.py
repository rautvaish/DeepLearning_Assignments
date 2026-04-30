import ultralytics
from ultralytics import YOLO
import os
import torch

def main():
    print("🚀 Initializing YOLOv11 Nano for Face Detection Training 🚀")
    
    # Check System Configuration
    print("\n--- System Configuration ---")
    if torch.cuda.is_available():
        device = "cuda:0"
        print(f"✅ GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠️ No GPU detected. Using CPU for training.")
        
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Ultralytics Version: {ultralytics.__version__}")
    print("----------------------------")
    
    # Define paths
    dataset_yaml = r"c:\Users\prana\OneDrive\Desktop\dl practicle 3\Face Detection.v27i.yolov11\data.yaml"
    
    # 1. Load a pretrained YOLOv11 nano model
    print("\n--- 1. Loading Model ---")
    model = YOLO("yolo11n.pt")
    
    # 2. Train the model
    # Note: We use 10 epochs for demonstration purposes. You can increase this for better accuracy.
    print("\n--- 2. Starting Training ---")
    results = model.train(
        data=dataset_yaml, 
        epochs=10,        # Number of training epochs
        imgsz=640,        # Image size 
        batch=16,         # Batch size
        device=device,    # Dynamically selected device (CPU/GPU)
        project="Face_Detection_Results", 
        name="yolo11n_training"
    )
    
    print("\n✅ Training Complete. Output saved to 'Face_Detection_Results/yolo11n_training'")
    
    # 3. Validation / Testing
    print("\n--- 3. Running Validation ---")
    metrics = model.val()  # Evaluate model performance on the validation set
    
    # 4. Inference / Output Results
    # Run prediction on a test image (if available) or validation image
    print("\n--- 4. Running Inference on Test Images ---")
    test_dir = r"c:\Users\prana\OneDrive\Desktop\dl practicle 3\Face Detection.v27i.yolov11\test\images"
    
    if os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0:
        test_images = [os.path.join(test_dir, f) for f in os.listdir(test_dir)][:5]  # Select first 5 images
        
        for img_path in test_images:
            predict_results = model.predict(
                source=img_path,
                conf=0.1,  # Lower confidence threshold so boxes appear even if the model is unsure
                save=True,
                project="Face_Detection_Results",
                name="yolo11n_inference"
            )
        print("\n✅ Inference Complete. Result images saved to 'Face_Detection_Results/yolo11n_inference'")
    else:
        print("\n⚠️ No test images found in the dataset folder, skipping inference step.")
        
    print("\n🎉 Object Detection Task (Part A) Successfully Completed! 🎉")

if __name__ == "__main__":
    main()
