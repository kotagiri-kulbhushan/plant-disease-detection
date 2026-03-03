import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
import gdown
from PIL import Image

st.set_page_config(page_title="Plant Disease Detection", layout="centered")

# ======================
# Download Model
# ======================
MODEL_PATH = "trained_model.keras"
FILE_ID = "13I2TotbKMvTjrOmKDTD6PlBa3zik3OS-"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model_from_drive()

# ======================
# Load class names
# ======================
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ======================
# UI
# ======================
st.title("🌿 AI Plant Disease Detection")
st.write("Upload a plant leaf image to detect disease.")

uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    top5_idx = prediction.argsort()[-5:][::-1]

    st.subheader("Top 5 Predictions")

    for i in top5_idx:
        st.write(f"{class_names[i]} — {round(prediction[i]*100,2)}%")
