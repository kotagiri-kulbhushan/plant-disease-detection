import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
import gdown
from PIL import Image
from datetime import datetime
import tempfile

# PDF imports
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4


# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Plant Health Diagnostic System",
    page_icon="🌿",
    layout="centered"
)

# =====================================================
# HEADER SECTION
# =====================================================
st.markdown("""
<h1 style='text-align: center; color: #2E7D32;'>
Plant Health Diagnostic System
</h1>
<p style='text-align: center; color: gray;'>
AI-powered Leaf Disease Detection for Farmers & Researchers
</p>
<hr>
""", unsafe_allow_html=True)


# =====================================================
# MODEL CONFIGURATION
# =====================================================
MODEL_PATH = "trained_model.keras"
FILE_ID = "13I2TotbKMvTjrOmKDTD6PlBa3zik3OS-"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Preparing AI model... Please wait."):
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# Load class labels
with open("class_names.json", "r") as f:
    class_names = json.load(f)


# =====================================================
# PDF REPORT GENERATOR
# =====================================================
def generate_pdf(image, disease, confidence, top5_results):

    pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(pdf_file.name, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>Plant Health Diagnostic Report</b>", styles["Title"]))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph(
        f"Report Generated On: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 20))

    # Save image temporarily
    img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    image.save(img_path)

    elements.append(RLImage(img_path, width=3*inch, height=3*inch))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"<b>Primary Diagnosis:</b> {disease}", styles["Heading2"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"<b>Model Confidence:</b> {confidence}%", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Top 5 Predictions
    data = [["Rank", "Condition", "Confidence (%)"]]
    for idx, (label, conf) in enumerate(top5_results, start=1):
        data.append([idx, label, conf])

    table = Table(data, colWidths=[1*inch, 3*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))

    elements.append(Paragraph("<b>Top 5 Model Predictions</b>", styles["Heading3"]))
    elements.append(Spacer(1, 10))
    elements.append(table)

    elements.append(Spacer(1, 30))
    elements.append(Paragraph(
        "Note: This report is AI-generated and should be verified by an agricultural expert if required.",
        styles["Normal"]
    ))

    doc.build(elements)
    return pdf_file.name


# =====================================================
# IMAGE UPLOAD SECTION
# =====================================================
st.subheader("Leaf Image Upload")

uploaded_file = st.file_uploader(
    "Upload a clear image of the affected leaf",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_container_width=True)

    if st.button("Run AI Diagnosis"):

        with st.spinner("Analyzing plant health condition..."):

            img = image.resize((128, 128))
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)[0]
            top5_idx = prediction.argsort()[-5:][::-1]

            main_disease = class_names[top5_idx[0]]
            main_confidence = round(prediction[top5_idx[0]] * 100, 2)

            top5_results = [
                (class_names[i], round(prediction[i] * 100, 2))
                for i in top5_idx
            ]

        # =====================================================
        # RESULT DISPLAY
        # =====================================================
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Diagnosis Result")

        if "healthy" in main_disease.lower():
            st.success(f"Status: Healthy Leaf")
        else:
            st.error(f"Status: Disease Detected")

        st.markdown(f"""
        **Identified Condition:** {main_disease}  
        **Confidence Level:** {main_confidence}%
        """)

        st.subheader("Model Confidence Distribution")

        for label, conf in top5_results:
            st.progress(conf / 100)
            st.write(f"{label} — {conf}%")

        # =====================================================
        # PDF DOWNLOAD
        # =====================================================
        pdf_path = generate_pdf(
            image,
            main_disease,
            main_confidence,
            top5_results
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Download Diagnostic Report (PDF)",
                data=f,
                file_name="Plant_Health_Report.pdf",
                mime="application/pdf"
            )

st.markdown("<hr>", unsafe_allow_html=True)
st.caption("© 2026 Plant Health AI System | Designed for Real-World Agricultural Use")[i]} — {round(prediction[i]*100,2)}%")
