from ultralytics import YOLO

def live_webcam_test():
    print("🎥 Starting Live Face Detection via Webcam 🎥")
    
    # 1. Load your newly trained custom model weights
    # The 'best.pt' file contains the optimal weights found during training
    custom_model_weights = "face_detection_best.pt"
    
    try:
        model = YOLO(custom_model_weights)
        print("✅ Custom model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 2. Run inference on the webcam (source=0 is the default laptop webcam)
    print("Press 'q' in the camera window to exit.")
    model.predict(
        source=0,       # 0 for webcam
        show=True,      # Displays a live window with bounding boxes
        conf=0.25       # Confidence threshold 
    )

if __name__ == "__main__":
    live_webcam_test()
