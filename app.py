import os
import numpy as np
import base64
import cv2
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load trained CNN model
MODEL_PATH = "Digit-Model.h5"
model = load_model(MODEL_PATH)

def preprocess_and_segment(image):
    """
    Preprocesses the image and extracts individual digit contours.
    Returns a list of digit images.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Apply morphological operations to separate digits
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)  

    # Find contours of digits
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    digit_images = []
    bounding_boxes = [cv2.boundingRect(c) for c in contours]

    # Filter out very small contours (noise)
    bounding_boxes = [b for b in bounding_boxes if b[2] > 10 and b[3] > 10]

    # Sort bounding boxes left to right
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])

    for x, y, w, h in bounding_boxes:
        digit = gray[y:y+h, x:x+w]

        # Ensure digit is properly padded and centered
        height, width = digit.shape
        padding = abs(height - width) // 2

        if height > width:
            digit = cv2.copyMakeBorder(digit, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=0)
        else:
            digit = cv2.copyMakeBorder(digit, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=0)

        # Resize to 28x28 for model input
        digit = cv2.resize(digit, (28, 28))

        # Normalize pixel values
        digit = digit.astype("float32") / 255.0
        digit = np.expand_dims(digit, axis=-1)  # (28,28,1)
        digit = np.expand_dims(digit, axis=0)   # (1,28,28,1)

        digit_images.append(digit)

    return digit_images


@app.route("/")
def home():
	return render_template('home.html')
    
@app.route("/login")
def login():
	return render_template('login.html')

@app.route("/performance")
def performance():
	return render_template('performance.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get base64 image from canvas
            canvas_img = request.form['canvasimg']
            img_data = base64.b64decode(canvas_img.split(',')[1])

            # Convert to OpenCV image
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # Process image and extract digits
            digit_images = preprocess_and_segment(img)

            # Predict digits
            predicted_digits = []
            for digit in digit_images:
                prediction = model.predict(digit)
                predicted_digits.append(str(np.argmax(prediction)))  # Get most probable digit

            return jsonify({"prediction": "".join(predicted_digits)})

        except Exception as e:
            return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
