from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
import json
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# ======================
# Base Directory (Important for Render)
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================
# Configuration
# ======================
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ======================
# Model Configuration
# ======================
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.keras")

FILE_ID = "13I2TotbKMvTjrOmKDTD6PlBa3zik3OS-"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model Loaded Successfully")

# ======================
# Load class names
# ======================
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

print("Total Classes:", len(class_names))


# ======================
# Prediction Function
# ======================
def predict_disease(img_path):
    img = image.load_img(img_path, target_size=(128, 128), color_mode='rgb')
    img_array = image.img_to_array(img)
    img_array = np.array(img_array, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]
    top5_idx = prediction.argsort()[-5:][::-1]

    top5_results = []
    for i in top5_idx:
        label = class_names[i]
        confidence = round(prediction[i] * 100, 2)
        top5_results.append((label, confidence))

    final_label, final_confidence = top5_results[0]
    return final_label, final_confidence, top5_results


# ======================
# Routes
# ======================
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    disease, confidence, top5_results = predict_disease(filepath)

    return render_template(
        'result.html',
        disease=disease,
        confidence=confidence,
        image_name=file.filename,
        top5_results=top5_results
    )


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ======================
# Run App
# ======================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
