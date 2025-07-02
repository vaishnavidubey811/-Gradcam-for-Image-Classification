from flask import Flask, render_template, request, jsonify, send_file
import os
import base64
import io
from PIL import Image
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from gradcam import visualize_gradcam, compute_gradcam, get_gradcam_model
import uuid
import tempfile

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  

model = None

def load_model():
    """Load the model once when the app starts"""
    global model
    if model is None:
        model_path = "model.h5"
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path, compile=False)
            print("Model loaded successfully!")
        else:
            print("Error: Model file not found!")
    return model

def save_gradcam_plot(img_path, model, target_size=(128, 128)):
    """Generate and save Grad-CAM visualization"""
    try:
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
        plt.title('Original X-ray', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap)
        plt.title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        class_label = "Pneumonia" if pred_class == 1 else "Normal"
        plt.imshow(superimposed_img)
        plt.title(f'Overlay\n{class_label} ({pred_confidence:.2%})', fontsize=12, fontweight='bold')
        plt.axis('off')
        
        plt.tight_layout()
        

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        return img_buffer, pred_class, pred_confidence, class_label
        
    except Exception as e:
        print(f"Error in save_gradcam_plot: {str(e)}")
        return None, None, None, None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and Grad-CAM analysis"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
        if not file.filename.lower().endswith(tuple('.' + ext for ext in allowed_extensions)):
            return jsonify({'error': 'Invalid file type. Please upload an image file.'}), 400
        
        model = load_model()
        if model is None:
            return jsonify({'error': 'Model not available'}), 500
        
        
        temp_dir = tempfile.gettempdir()
        temp_filename = f"upload_{uuid.uuid4().hex}.jpg"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        img = Image.open(file.stream)
        img = img.convert('RGB')
        img.save(temp_path, 'JPEG')
        
        
        img_buffer, pred_class, pred_confidence, class_label = save_gradcam_plot(temp_path, model)
        
        if img_buffer is None:
            return jsonify({'error': 'Failed to process image'}), 500
        
        
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'prediction_class': int(pred_class),
            'confidence': f"{pred_confidence:.2%}",
            'class_label': class_label,
            'filename': file.filename
        })
        
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/path_analysis', methods=['POST'])
def analyze_by_path():
    """Handle image path analysis"""
    try:
        data = request.get_json()
        image_path = data.get('image_path', '').strip()
        
        if not image_path:
            return jsonify({'error': 'No image path provided'}), 400
        
        if not os.path.exists(image_path):
            return jsonify({'error': f'File not found: {image_path}'}), 400
        
        allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
        if not image_path.lower().endswith(tuple('.' + ext for ext in allowed_extensions)):
            return jsonify({'error': 'Invalid file type. Please provide an image file path.'}), 400
        
        model = load_model()
        if model is None:
            return jsonify({'error': 'Model not available'}), 500
        

        img_buffer, pred_class, pred_confidence, class_label = save_gradcam_plot(image_path, model)
        
        if img_buffer is None:
            return jsonify({'error': 'Failed to process image'}), 500
        
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'prediction_class': int(pred_class),
            'confidence': f"{pred_confidence:.2%}",
            'class_label': class_label,
            'filename': os.path.basename(image_path)
        })
        
    except Exception as e:
        print(f"Error in analyze_by_path: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000) 