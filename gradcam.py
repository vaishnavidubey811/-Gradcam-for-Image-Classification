import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import os
import sys

def get_gradcam_model(model):
    """Create a model that outputs the last convolutional layer and the prediction"""
    last_conv_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
    
    if last_conv_layer is None:
        raise ValueError("No convolutional layer found in the model")
    
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[last_conv_layer.output, model.output]
    )
    return grad_model, last_conv_layer.name

def compute_gradcam(model, img_array, layer_name, pred_index=None):
    """Compute Grad-CAM heatmap for the given image"""
    grad_model, _ = get_gradcam_model(model)
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        conv_outputs[:, :, i] *= pooled_grads[i]
    

    heatmap = np.mean(conv_outputs, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    return heatmap

def visualize_gradcam(img_path, model, target_size=(128, 128)):
    """Visualize Grad-CAM heatmap on the input image"""
    
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = img.copy()
    
    
    img = cv2.resize(img, target_size)
    img_array = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
    
    
    preds = model.predict(img_array)
    pred_class = np.argmax(preds[0])
    pred_confidence = preds[0][pred_class]
    
    
    heatmap = compute_gradcam(model, img_array, None, pred_class)
    

    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    

    superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)
    

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title('Original X-ray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title(f'Overlay (Prediction: {pred_class}, Confidence: {pred_confidence:.2%})')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return pred_class, pred_confidence

def get_user_input():
    """
    Get image path from user input
    """
    print("\n" + "="*60)
    print("GRAD-CAM CHEST X-RAY ANALYSIS")
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
    Main function to run the interactive Grad-CAM analysis
    """
    model_path = "model.h5"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        print("Please make sure the model.h5 file is in the current directory.")
        return
    
    print("Loading model...")
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully!")
        
        image_path = get_user_input()
        
        print(f"\nAnalyzing image: {image_path}")
        print("Please wait...")
        
        
        pred_class, confidence = visualize_gradcam(image_path, model)
        
        print(f"\n" + "="*60)
        print("GRAD-CAM ANALYSIS RESULTS")
        print("="*60)
        print(f"Image: {image_path}")
        print(f"Prediction Class: {pred_class}")
        print(f"Confidence: {confidence:.2%}")
        print("="*60)
        
        print("\nThank you for using the Grad-CAM Analysis tool!")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        print("Please try again with a different image.")

if __name__ == "__main__":
    main() 