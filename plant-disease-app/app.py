import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
import gdown
from PIL import Image
from datetime import datetime
import tempfile

# PDF
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Plant Disease Detection System",
    page_icon="🌿",
    layout="centered"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================
# CUSTOM CSS (UI POLISH)
# =====================================================
st.markdown("""
<style>
.uploaded-img {
    border-radius: 20px;
    width: 300px;
    margin: auto;
    display: block;
}
.center-btn {
    display: flex;
    justify-content: center;
}
.summary-title {
    text-align: center;
    font-size: 32px;
    font-weight: 700;
    margin-top: 30px;
}
.result-card {
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 25px;
    border-radius: 15px;
    color: white;
    margin-top: 20px;
}
.table-container {
    background: #f5f7fa;
    padding: 20px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown("<h1 style='text-align:center;'>🌿 Plant Disease Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>AI-powered Leaf Diagnosis for Agriculture</p>", unsafe_allow_html=True)
st.markdown("---")

# =====================================================
# MODEL
# =====================================================
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.keras")
FILE_ID = "13I2TotbKMvTjrOmKDTD6PlBa3zik3OS-"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(DOWNLOAD_URL, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

with open(os.path.join(BASE_DIR, "class_names.json"), "r") as f:
    class_names = json.load(f)

# =====================================================
# PDF FUNCTION (PROFESSIONAL)
# =====================================================
def generate_pdf(image, disease, confidence, top5_results):

    pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(pdf_file.name, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    center_style = ParagraphStyle(
        name="Center",
        parent=styles["Title"],
        alignment=1
    )

    elements.append(Paragraph("🌿 Plant Disease Detection System", center_style))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Detection Report", center_style))
    elements.append(Spacer(1, 20))

    # Save image
    img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    image.save(img_path)
    elements.append(RLImage(img_path, width=3*inch, height=3*inch))
    elements.append(Spacer(1, 20))

    clean = disease.replace("___", " - ")

    elements.append(Paragraph("<b>Detected Disease</b>", styles["Heading2"]))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(clean, styles["Heading3"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"<b>Model Confidence:</b> {confidence}%", styles["Normal"]))
    elements.append(Spacer(1, 25))

    # Table
    data = [["Rank", "Disease", "Confidence"]]
    for i, (label, conf) in enumerate(top5_results, 1):
        data.append([i, label.replace("___", " - "), f"{conf}%"])

    table = Table(data, colWidths=[1*inch, 3*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#667eea")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 10),
    ]))

    elements.append(Paragraph("<b>Top 5 Predictions</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    elements.append(table)

    elements.append(Spacer(1, 30))
    elements.append(Paragraph(
        f"Generated on {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        styles["Normal"]
    ))

    doc.build(elements)
    return pdf_file.name

# =====================================================
# UPLOAD
# =====================================================
st.subheader("Upload Leaf Image")

uploaded = st.file_uploader("Choose image", type=["jpg","jpeg","png"])

if uploaded:
    image = Image.open(uploaded)
    st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
    st.image(image, width=300)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='center-btn'>", unsafe_allow_html=True)
    run = st.button("🔍 Run AI Diagnosis")
    st.markdown("</div>", unsafe_allow_html=True)

    if run:

        img = image.resize((128,128))
        arr = np.expand_dims(np.array(img), axis=0)
        pred = model.predict(arr)[0]
        idx = pred.argsort()[-5:][::-1]

        main = class_names[idx[0]]
        main_conf = round(float(pred[idx[0]])*100,2)

        top5 = [(class_names[i], round(float(pred[i])*100,2)) for i in idx]

        st.markdown("<div class='summary-title'>Diagnostic Summary</div>", unsafe_allow_html=True)

        clean_main = main.replace("___"," - ")

        st.markdown(f"""
        <div class='result-card'>
            <h3>Detected Disease</h3>
            <h2>{clean_main}</h2>
            <p>Confidence: <b>{main_conf}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Top 5 Predictions")

        st.markdown("<div class='table-container'>", unsafe_allow_html=True)

        st.markdown("""
        | Rank | Disease | Confidence |
        |------|---------|------------|
        """)

        for i,(label,conf) in enumerate(top5,1):
            st.markdown(f"| {i} | {label.replace('___',' - ')} | **{conf}%** |")

        st.markdown("</div>", unsafe_allow_html=True)

        pdf_path = generate_pdf(image, main, main_conf, top5)

        with open(pdf_path,"rb") as f:
            st.download_button(
                "📄 Download Professional Report",
                f,
                file_name="Plant_Disease_Report.pdf",
                mime="application/pdf"
            )

st.markdown("---")
st.caption("© 2026 Plant Disease Detection System | Agricultural AI")
