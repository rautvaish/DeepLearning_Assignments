from ultralytics import YOLO
import os

def test_classification():
    print("🎥 Running Classification Inference 🎥")
    
    model_path = "car_color_best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}. Please train first.")
        return

    model = YOLO(model_path)
    
    # Path to some test images
    test_img_dir = r"c:\Users\prana\OneDrive\Desktop\dl practicle 3\classification\test"
    
    # Get a list of subdirectories (classes) in test folder
    classes = [d for d in os.listdir(test_img_dir) if os.path.isdir(os.path.join(test_img_dir, d))]
    
    if not classes:
        print("❌ No test images found.")
        return

    # Pick one image from the first class for testing
    first_class = classes[0]
    class_dir = os.path.join(test_img_dir, first_class)
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if images:
        test_image = os.path.join(class_dir, images[0])
        print(f"Testing on image: {test_image} (Actual Class: {first_class})")
        
        results = model.predict(source=test_image, save=True, project="Car_Color_Results", name="test_inference")
        
        # Display the result (top 1 class)
        for result in results:
            top1_idx = result.probs.top1
            top1_name = result.names[top1_idx]
            top1_conf = result.probs.top1conf
            print(f"✅ Prediction: {top1_name} with {top1_conf:.2f} confidence")
    else:
        print("❌ No images found in test directories.")

if __name__ == "__main__":
    test_classification()
