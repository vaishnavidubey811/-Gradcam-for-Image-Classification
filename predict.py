import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt
import os
import sys

def preprocess_image(img_path, target_size=(128, 128)):
    """
    Preprocess the image for model prediction
    """
    try:
        
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image at {img_path}")
        
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        original_img = img.copy()
        
        img = cv2.resize(img, target_size)
        
        img = img.astype(np.float32) / 255.0
        
        img = np.expand_dims(img, axis=0)
        
        return img, original_img
    except Exception as e:
        print(f"Error preprocessing image: {str(e)}")
        return None, None

def predict_pneumonia(model_path, image_path, confidence_threshold=0.5):
    """
    Predict pneumonia from X-ray image with improved accuracy
    """
    try:
        
        model = load_model(model_path, compile=False)
        
        
        processed_img, original_img = preprocess_image(image_path)
        if processed_img is None:
            return "Error processing image", None
        
        prediction = model.predict(processed_img)
        
        
        normal_conf = prediction[0][0]
        pneumonia_conf = prediction[0][1]
        
        
        if pneumonia_conf > confidence_threshold:
            result = f"Pneumonia (Confidence: {pneumonia_conf:.2%})"
        elif normal_conf > confidence_threshold:
            result = f"Normal (Confidence: {normal_conf:.2%})"
        else:
            result = "Uncertain - Please consult a medical professional"
            
        return result, original_img
            
    except Exception as e:
        return f"Error during prediction: {str(e)}", None

def get_user_input():
    """
    Get image path from user input
    """
    print("\n" + "="*60)
    print("CHEST X-RAY PNEUMONIA DETECTION")
    print("="*60)
    
    print("\n" + "-"*60)
    
    while True:
        
        image_path = input("Enter the path to your X-ray image (or 'quit' to exit): ").strip()
        
        if image_path.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            sys.exit(0)
        
        
        if not os.path.exists(image_path):
            print(f"❌ Error: File '{image_path}' not found!")
            print("Please enter a valid file path.")
            continue
        
        
        if not image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            print(f"❌ Error: '{image_path}' is not a recognized image file!")
            print("Please use .png, .jpg, .jpeg, .bmp, or .tiff files.")
            continue
        
        return image_path

def main():
    """
    Main function to run the interactive prediction
    """
    model_path = "model.h5"
    
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        print("Please make sure the model.h5 file is in the current directory.")
        return
    
    print("Loading model...")
    
    while True:
        try:
            
            image_path = get_user_input()
            
            print(f"\nAnalyzing image: {image_path}")
            print("Please wait...")
            
            
            result, img = predict_pneumonia(model_path, image_path)
            
            print(f"\n" + "="*60)
            print("PREDICTION RESULTS")
            print("="*60)
            print(f"Image: {image_path}")
            print(f"Result: {result}")
            print("="*60)
            
            
            if img is not None:
                plt.figure(figsize=(12, 8))
                plt.imshow(img)
                plt.title(f"X-ray Image Analysis\n{result}", fontsize=14, fontweight='bold')
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            
            
            print("\nThank you for using the Chest X-ray Analysis tool!")
            break
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Please try again with a different image.")

if __name__ == "__main__":
    main() 