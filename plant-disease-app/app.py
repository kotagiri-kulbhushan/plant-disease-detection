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
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4


# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Plant Health Diagnostic System",
    page_icon="🌿",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<h1 style='text-align:center; color:#2E7D32;'>
Plant Health Diagnostic System
</h1>
<p style='text-align:center; text-align:center; color:gray;'>
AI-powered Leaf Disease Detection for Farmers & Researchers
</p>
<hr>
""", unsafe_allow_html=True)

# =====================================================
# MODEL CONFIG
# =====================================================
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.keras")
FILE_ID = "13I2TotbKMvTjrOmKDTD6PlBa3zik3OS-"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Preparing AI model..."):
            gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# =====================================================
# LOAD CLASS NAMES
# =====================================================
CLASS_PATH = os.path.join(BASE_DIR, "class_names.json")

with open(CLASS_PATH, "r") as f:
    class_names = json.load(f)

# =====================================================
# PDF GENERATOR
# =====================================================
def generate_pdf(image, disease, confidence, top5_results):

    pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(pdf_file.name, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    clean_disease = disease.replace("___", " - ")

    # Header
    elements.append(Paragraph("<b>Plant Health Diagnostic Report</b>", styles["Title"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%d %B %Y, %H:%M')}",
        styles["Normal"]
    ))
    elements.append(Spacer(1, 20))

    # Leaf Image
    img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    image.save(img_path)
    elements.append(RLImage(img_path, width=3.5*inch, height=3.5*inch))
    elements.append(Spacer(1, 20))

    # Diagnosis Section
    elements.append(Paragraph("<b>Primary Diagnosis</b>", styles["Heading2"]))
    elements.append(Spacer(1, 8))
    elements.append(Paragraph(clean_disease, styles["Heading3"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"Model Confidence: {confidence}%", styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Interpretation
    if confidence >= 90:
        interpretation = "High confidence prediction."
    elif confidence >= 70:
        interpretation = "Moderate confidence prediction."
    else:
        interpretation = "Low confidence prediction. Further verification recommended."

    elements.append(Paragraph("<b>Interpretation</b>", styles["Heading3"]))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(interpretation, styles["Normal"]))
    elements.append(Spacer(1, 20))

    # Top Predictions Table
    data = [["Rank", "Condition", "Confidence (%)"]]
    for idx, (label, conf) in enumerate(top5_results, start=1):
        label_clean = label.replace("___", " - ")
        data.append([idx, label_clean, conf])

    table = Table(data, colWidths=[1*inch, 3*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F5E9")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))

    elements.append(Paragraph("<b>Top 5 Model Predictions</b>", styles["Heading3"]))
    elements.append(Spacer(1, 10))
    elements.append(table)
    elements.append(Spacer(1, 30))

    elements.append(Paragraph(
        "Disclaimer: This report is AI-generated and should be validated by an agricultural expert.",
        styles["Normal"]
    ))

    doc.build(elements)
    return pdf_file.name


# =====================================================
# IMAGE UPLOAD
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
            main_confidence = round(float(prediction[top5_idx[0]]) * 100, 2)

            top5_results = [
                (class_names[i], round(float(prediction[i]) * 100, 2))
                for i in top5_idx
            ]

        # =====================================================
        # DISPLAY RESULTS
        # =====================================================
        st.markdown("---")
        st.subheader("Diagnostic Summary")

        clean_disease = main_disease.replace("___", " - ")

        # Confidence Interpretation
        if main_confidence >= 90:
            confidence_level = "High Confidence"
            confidence_color = "green"
        elif main_confidence >= 70:
            confidence_level = "Moderate Confidence"
            confidence_color = "orange"
        else:
            confidence_level = "Low Confidence"
            confidence_color = "red"

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Identified Condition")
            st.markdown(f"<h4 style='color:#2E7D32'>{clean_disease}</h4>", unsafe_allow_html=True)

        with col2:
            st.markdown("### Confidence Level")
            st.markdown(
                f"<h4 style='color:{confidence_color}'>{main_confidence}%</h4>",
                unsafe_allow_html=True
            )
            st.caption(confidence_level)

        if "healthy" in main_disease.lower():
            st.success("Leaf Status: Healthy")
        else:
            st.error("Leaf Status: Disease Detected")

        st.markdown("### Model Confidence Breakdown")

        for label, conf in top5_results:
            label_clean = label.replace("___", " - ")
            progress_value = int(round(conf))
            st.progress(progress_value)
            st.write(f"{label_clean} — {progress_value}%")

        st.markdown("---")

        # PDF DOWNLOAD
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

