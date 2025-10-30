import struct
import sqlite3
import io
import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from datetime import datetime
import requests

# ------------------------------
# Télécharger le modèle si absent
# ------------------------------
MODEL_PATH = 'models/system.h5'
MODEL_URL = "https://drive.google.com/file/d/1TQOytruN-z1UeRQDe8ylQBjfLrOs1_lI/view?usp=sharing"

def download_model_if_missing():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        st.info("Modèle téléchargé depuis Google Drive.")

download_model_if_missing()

# Charger le modèle
model = load_model(MODEL_PATH)

# ------------------------------
# Prétraitement et prédiction
# ------------------------------
def preprocess_image(image, target_size=(150, 150)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return float(prediction[0][0])

# ------------------------------
# Base de données SQLite
# ------------------------------
def init_db():
    conn = sqlite3.connect("db/results.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        image_name TEXT,
        result TEXT,
        probability REAL
        )
    """)
    conn.commit()
    conn.close()

def save_to_db(image_name, result, probability):
    conn = sqlite3.connect("db/results.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions (image_name, result, probability)
        VALUES (?, ?, ?)
    """, (str(image_name), str(result), float(probability)))
    conn.commit()
    conn.close()

# ------------------------------
# Génération du PDF
# ------------------------------
def create_pdf(session_results):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 60, "Rapport de Détection de Pneumonie")
    c.setFont("Helvetica", 11)
    c.drawString(50, height - 80, f"Date : {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    y = height - 110

    for idx, (filename, label, probability, image_bytes) in enumerate(session_results):
        if y < 200:
            c.showPage()
            y = height - 60

        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, f"{idx+1}. {filename}")
        y -= 18

        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_w, img_h = pil_img.size
            max_w = width - 120
            max_h = 300
            ratio = min(max_w / img_w, max_h / img_h, 1.0)
            disp_w = img_w * ratio
            disp_h = img_h * ratio
            img_reader = ImageReader(pil_img)
            c.drawImage(img_reader, 60, y - disp_h, width=disp_w, height=disp_h)
            y -= (disp_h + 10)
        except Exception as e:
            c.setFont("Helvetica", 10)
            c.drawString(60, y, f"[Impossible d'insérer l'image : {e}]")
            y -= 18

        c.setFont("Helvetica", 11)
        c.drawString(60, y, f"Etat : {label}")
        y -= 16
        c.drawString(60, y, f"Probabilité : {probability:.2f}")
        y -= 26

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# ------------------------------
# Interface utilisateur Streamlit
# ------------------------------
st.title("Détection de la Pneumonie dans les images radiographiques à l'aide de l'intelligence artificielle !")
st.write("Upload one or more x-ray images to detect if they show signs of pneumonia.")

init_db()

uploaded_files = st.file_uploader("Choose one or more images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

session_results_for_pdf = []

if uploaded_files:
    results = []
    session_results_for_pdf = []
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        try:
            image = Image.open(io.BytesIO(file_bytes))
        except Exception as e:
            st.error(f"Impossible d'ouvrir {uploaded_file.name} : {e}")
            continue

        st.image(image, caption=f"Image : {uploaded_file.name}", use_container_width=True)

        result = predict(image)
        label = "Pneumonia detected" if result > 0.5 else "No signs of pneumonia"
        probability = result
        results.append((uploaded_file.name, label, probability))
        save_to_db(uploaded_file.name, label, probability)
        session_results_for_pdf.append((uploaded_file.name, label, probability, file_bytes))

        if result > 0.5:
            st.error(f"Result for {uploaded_file.name} : {label} (probability : {result:.2f})")
        else:
            st.success(f"Result for {uploaded_file.name} : {label} (probability : {result:.2f})")

    # Statistiques globales
    st.write("### Global Statistics")
    total_images = len(results)
    pneumonia_detected = sum(1 for _, label, _ in results if label == "Pneumonia detected")
    st.write(f"Total number of images : {total_images}")
    st.write(f"Number of images with pneumonia detected : {pneumonia_detected}")
    st.write(f"Percentage of pneumonia detected : {(pneumonia_detected / total_images) *100:.2f}%")

    # Pie chart
    labels_chart = ['No signs of pneumonia', 'Pneumonia detected']
    values_chart = [total_images - pneumonia_detected, pneumonia_detected]
    fig, ax = plt.subplots()
    ax.pie(values_chart, labels=labels_chart, autopct='%1.1f%%', startangle=90, colors=['#1f77b4', '#ff7f0e'])
    ax.axis('equal')
    st.pyplot(fig)

    # PDF
    if session_results_for_pdf:
        pdf_bytes = create_pdf(session_results_for_pdf)
        pdf_filename = f"pneumonia_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.pdf"

        # Bouton violet centré
        st.markdown(
            """
            <style>
            .stDownloadButton>button {
                background-color: #6a0dad !important;
                color: white !important;
                border-radius: 8px !important;
                padding: 8px 16px !important;
                font-weight: 600 !important;
            }
            </style>
            """, unsafe_allow_html=True
        )
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.download_button(
                label="Télécharger le rapport PDF",
                data=pdf_bytes,
                file_name=pdf_filename,
                mime="application/pdf"
            )

# ------------------------------
# Afficher les résultats enregistrés
# ------------------------------
st.write("### Recorded Results")
if st.button("Show results"):
    conn = sqlite3.connect("db/results.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM predictions")
    rows = cursor.fetchall()
    conn.close()

    if rows:
        st.write("### Results Details")
        for row in rows:
            image_name = row[1].decode('utf-8') if isinstance(row[1], bytes) else row[1]
            result = row[2].decode('utf-8') if isinstance(row[2], bytes) else row[2]
            if isinstance(row[3], bytes):
                try:
                    probability = struct.unpack('f', row[3])[0]
                except:
                    probability = float(np.frombuffer(row[3], dtype=np.float32)[0])
            else:
                probability = float(row[3])
            st.write(f"Image : {image_name} | Result : {result} | probability : {probability:.2f}")
    else:
        st.write("No results recorded.")
