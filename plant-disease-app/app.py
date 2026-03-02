from flask import Flask, render_template, request, send_from_directory
import os
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# ======================
# Configuration
# ======================
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = r"D:\plant disease detection\trained_model.keras"

# Load model
model = load_model(MODEL_PATH)
print("Model Loaded")

# ======================
# Load class names (exact order from Colab)
# ======================
with open("class_names.json", "r") as f:
    class_names = json.load(f)

print("Total Classes:", len(class_names))


# ======================
# Prediction Function
# ======================
def predict_disease(img_path):
    # Load image (same as training)
    img = image.load_img(img_path, target_size=(128, 128), color_mode='rgb')
    img_array = image.img_to_array(img)

    # IMPORTANT: No normalization (training used 0–255)
    img_array = np.array(img_array, dtype=np.float32)

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array, verbose=0)[0]

    # Top 5 predictions
    top5_idx = prediction.argsort()[-5:][::-1]

    top5_results = []
    for i in top5_idx:
        label = class_names[i]
        confidence = round(prediction[i] * 100, 2)
        top5_results.append((label, confidence))

    # Main prediction
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

    # Save uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Predict
    disease, confidence, top5_results = predict_disease(filepath)

    return render_template(
        'result.html',
        disease=disease,
        confidence=confidence,
        image_name=file.filename,
        top5_results=top5_results
    )


# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# ======================
# Run App
# ======================
if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)