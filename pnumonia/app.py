import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Set upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
MODEL_PATH = r"D:\AI Project\pnumonia\pnumonia\pnumonia.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Model expects image size 150x150
TARGET_SIZE = (150, 150)

# Define class labels
CLASS_LABELS = ['Normal','Pneumonia']  # Ensure correct order

def preprocess_image(image_path):
    """ Load and preprocess image for model prediction """
    img = image.load_img(image_path, target_size=TARGET_SIZE)  # Resize correctly
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dims for batch
    img_array = img_array / 255.0  # Normalize
    return img_array

def predict_image(image_path):
    """ Perform model prediction """
    try:
        # Preprocess image
        img_array = preprocess_image(image_path)

        # Debugging info
        print(f"✅ Image Shape Before Prediction: {img_array.shape}")

        # Make prediction
        predictions = model.predict(img_array)

        # Debugging output
        print(f"🔍 Raw Model Output: {predictions}")

        # Handle binary or multi-class classification
        if predictions.shape[1] == 1:  # Binary classification (sigmoid activation)
            probability = float(predictions[0][0])
            predicted_class_index = int(probability > 0.5)  # Threshold 0.5
        else:  # Multi-class classification (softmax activation)
            predicted_class_index = np.argmax(predictions)
            probability = float(predictions[0][predicted_class_index])

        predicted_class = CLASS_LABELS[predicted_class_index]
        return predicted_class, probability

    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        return "Prediction failed", None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    """ Handle image upload and prediction """
    if request.method == 'POST':
        # Check if file is present in request
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")

        file = request.files['file']

        # Ensure a file was actually uploaded
        if file.filename == '':
            return render_template('index.html', error="No file selected")

        # Save file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"✅ Image saved: {file_path}")

        # Predict
        predicted_class, probability = predict_image(file_path)

        # Handle errors
        if probability is None:
            return render_template('index.html', error="Prediction failed")

        result_text = f"Predicted Class: {predicted_class} (Confidence: {probability:.4f})"
        return render_template('index.html', filename=filename, result=result_text)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
