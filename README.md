# Grad-CAM Project for Chest X-ray Analysis

This project implements Grad-CAM (Gradient-weighted Class Activation Mapping) for analyzing chest X-ray images to detect pneumonia.

## Setup Instructions

### 1. Virtual Environment Setup

The project is already set up with a virtual environment. To activate it:

Windows:
```bash
grad_cam_env\Scripts\activate
```

Linux/Mac:
```bash
source grad_cam_env/bin/activate
```

### 2. Dependencies

All required dependencies are already installed:
- TensorFlow 2.19.0
- OpenCV 4.11.0
- Matplotlib 3.10.3
- NumPy 2.1.3
- Flask 3.1.1
- Pillow (PIL)


## Usage

###  Web Application 

The easiest way to use the project is through the web interface:

1. Start the web application:
   ```bash
   python app.py
   ```

2. Open your browser and go to:
   ```
   http://localhost:5000
   ```

3. Use the web interface to:
   - Upload X-ray images directly
   - Enter image paths for analysis
   - View Grad-CAM visualizations
   - See prediction results

###  Web Interface Features

- **File Upload**: Drag and drop or select X-ray images
- **Path Input**: Enter the path to any image file
- **Real-time Analysis**: Get results instantly
- **Visual Results**: See original image, heatmap, and overlay
- **Prediction Details**: View confidence scores and classifications

###  Command Line Tools

#### Running Predictions

To predict pneumonia from an X-ray image:
```bash
python predict.py
```

This will analyze the image specified in the script (currently set to `img3.jpeg`).

#### Running Grad-CAM Analysis

To run Grad-CAM analysis on a single image:
```bash
python gradcam.py
```

This will prompt you to enter an image path and show the Grad-CAM visualization.


## Project Structure

- `app.py` - Flask web application
- `templates/index.html` - Web interface template
- `gradcam.py` - Core Grad-CAM implementation
- `predict.py` - Simple prediction script
- `model.h5` - Trained model for chest X-ray classification
- `uploads/` - Directory containing test images
- `chest_xray/` - Dataset directory
- `model_weights/` - Additional model weights
- `requirements.txt` - Python dependencies

## Model Information

The model is trained to classify chest X-ray images into two categories:
- Normal
- Pneumonia

The model achieves high accuracy in detecting pneumonia from chest X-ray images.
