from ultralytics import YOLO
import os

def test_pose():
    print("🎥 Running Pose Estimation Inference 🎥")
    
    model_path = "yoga_pose_best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}. Please train first.")
        return

    model = YOLO(model_path)
    
    # Path to some test images
    test_img_dir = r"c:\Users\prana\OneDrive\Desktop\dl practicle 3\Pose Estimation\Yoga Pose.v2i.yolov8\test\images"
    
    if os.path.exists(test_img_dir) and len(os.listdir(test_img_dir)) > 0:
        test_images = [os.path.join(test_img_dir, f) for f in os.listdir(test_img_dir)][:3]
        
        for img_path in test_images:
            print(f"Testing on image: {img_path}")
            results = model.predict(source=img_path, save=True, project="Yoga_Pose_Results", name="test_inference", show=False)
            print(f"✅ Keypoints detected: {len(results[0].keypoints) if results[0].keypoints else 0}")
    else:
        print("❌ No test images found.")

if __name__ == "__main__":
    test_pose()
