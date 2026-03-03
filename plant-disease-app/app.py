import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
import gdown
from PIL import Image, ImageDraw
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
# HEADER
# =====================================================
st.markdown(
    "<h1 style='text-align:center;'>Plant Disease Detection System</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:gray;'>AI-powered Leaf Diagnosis</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# =====================================================
# MODEL LOADING
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
# ROUND IMAGE FUNCTION FOR PDF
# =====================================================
def make_rounded_image(img, radius=40):
    img = img.convert("RGBA")
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        [(0, 0), img.size],
        radius=radius,
        fill=255
    )
    img.putalpha(mask)
    return img


# =====================================================
# PDF GENERATION (PROFESSIONAL STRUCTURE)
# =====================================================
def generate_pdf(image, disease, confidence, top5):

    pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    doc = SimpleDocTemplate(
        pdf_file.name,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    elements = []
    styles = getSampleStyleSheet()

    # Custom styles
    center_title = ParagraphStyle(
        name="CenterTitle",
        parent=styles["Title"],
        alignment=1,
        fontSize=20
    )

    right_align = ParagraphStyle(
        name="RightAlign",
        parent=styles["Normal"],
        alignment=2
    )

    # Date at top right
    elements.append(
        Paragraph(
            datetime.now().strftime("%d %B %Y  |  %H:%M"),
            right_align
        )
    )
    elements.append(Spacer(1, 10))

    # Title centered
    elements.append(Paragraph("Plant Disease Detection Report", center_title))
    elements.append(Spacer(1, 25))

    # Rounded image
    rounded = make_rounded_image(image)
    img_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
    rounded.save(img_path, format="PNG")

    elements.append(RLImage(img_path, width=3*inch, height=3*inch))
    elements.append(Spacer(1, 25))

    clean = disease.replace("___", " - ")

    # Highlighted Diagnosis Box
    diagnosis_data = [
        ["Detected Disease", clean],
        ["Confidence", f"{confidence}%"]
    ]

    diag_table = Table(diagnosis_data, colWidths=[2.5*inch, 2.5*inch])
    diag_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#E0E7FF")),
        ("TEXTCOLOR", (0,0), (-1,-1), colors.black),
        ("FONTNAME", (0,0), (-1,-1), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,-1), 12),
        ("ALIGN", (1,0), (1,-1), "RIGHT"),
        ("INNERGRID", (0,0), (-1,-1), 0.5, colors.white),
        ("BOX", (0,0), (-1,-1), 1, colors.HexColor("#C7D2FE")),
        ("LEFTPADDING", (0,0), (-1,-1), 12),
        ("RIGHTPADDING", (0,0), (-1,-1), 12),
        ("TOPPADDING", (0,0), (-1,-1), 12),
        ("BOTTOMPADDING", (0,0), (-1,-1), 12),
    ]))

    elements.append(diag_table)
    elements.append(Spacer(1, 30))

    # Top 5 Section
    elements.append(Paragraph("Top 5 Model Predictions", styles["Heading2"]))
    elements.append(Spacer(1, 12))

    data = [["Rank", "Disease", "Confidence"]]
    for i, (label, conf) in enumerate(top5, 1):
        data.append([i, label.replace("___", " - "), f"{conf}%"])

    table = Table(data, colWidths=[1*inch, 3*inch, 1.2*inch])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#6366F1")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("ALIGN", (2,1), (2,-1), "RIGHT"),
        ("FONTSIZE", (0,0), (-1,-1), 10),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 30))

    # Disclaimer
    elements.append(
        Paragraph(
            "<i>Disclaimer: This report is AI-generated and should be validated by an agricultural expert.</i>",
            styles["Normal"]
        )
    )

    doc.build(elements)
    return pdf_file.name


# =====================================================
# IMAGE UPLOAD UI
# =====================================================
st.subheader("Upload Leaf Image")

uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(image, width=280)

    st.markdown("")

    colb1, colb2, colb3 = st.columns([1,2,1])
    with colb2:
        run = st.button("Run AI Diagnosis", use_container_width=True)

    if run:
        img = image.resize((128,128))
        arr = np.expand_dims(np.array(img), axis=0)
        pred = model.predict(arr)[0]
        idx = pred.argsort()[-5:][::-1]

        main = class_names[idx[0]]
        main_conf = round(float(pred[idx[0]])*100,2)
        top5 = [(class_names[i], round(float(pred[i])*100,2)) for i in idx]

        clean_main = main.replace("___"," - ")

        st.markdown("---")
        st.markdown(
            "<h2 style='text-align:center;'>Diagnostic Summary</h2>",
            unsafe_allow_html=True
        )

        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg,#6366F1,#8B5CF6);
            padding:25px;
            border-radius:20px;
            color:white;
            margin-top:20px;">
            <h3>Detected Disease</h3>
            <h2>{clean_main}</h2>
            <p style='font-size:18px;'>Confidence: <b>{main_conf}%</b></p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Top 5 Predictions")

        for i,(label,conf) in enumerate(top5,1):
            cols = st.columns([1,4,2])
            cols[0].write(i)
            cols[1].write(label.replace("___"," - "))
            cols[2].write(f"{conf}%")

        pdf_path = generate_pdf(image, main, main_conf, top5)

        with open(pdf_path,"rb") as f:
            st.download_button(
                "Download Professional Report",
                f,
                file_name="Plant_Disease_Report.pdf",
                mime="application/pdf"
            )
