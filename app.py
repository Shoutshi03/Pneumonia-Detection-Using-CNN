import struct
import sqlite3
import numpy as np
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Charger le modèle
MODEL_PATH = 'models/system.h5'  # Remplacez par le chemin de votre modèle
model = load_model(MODEL_PATH)

# Prétraitement de l'image
def preprocess_image(image, target_size=(150, 150)):
    # Convertir l'image en RGB si elle est en niveaux de gris
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
    return prediction[0][0]

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
    
# Interface utilisateur
st.title("Pneumonie Detection using Deep Convolutional neural network !")
st.write("Upload one or more x-ray images to detect if they show signs of pneumonia.")

init_db()

uploaded_files = st.file_uploader("Choose one or more images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    results = []
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Image : {uploaded_file.name}", use_container_width=True)
        
        result = predict(image)
        label = "Pneumonia detected" if result > 0.5 else "No signs of pneumonia"
        probability = result
        results.append((uploaded_file.name, label, probability))
        save_to_db(uploaded_file.name, label, probability)
    
        if result > 0.5:
            st.error(f"Result for {uploaded_file.name} : {label} (probability : {result:.2f})")
        else:
            st.success(f"Result for {uploaded_file.name} : {label} (probability : {result:.2f})")
     
    st.write("### Global Statistics")
    total_images = len(results)
    pneumonia_detected = sum(1 for _, label, _ in results if label == "Pneumonia detected")
    st.write(f"Total number of images : {total_images}")
    st.write(f"Number of images with pneumonia detected : {pneumonia_detected}")
    st.write(f"Percentage of pneumonia detected : {(pneumonia_detected / total_images) *100:.2f}%")
    st.write("### Visualization of Results")
    labels = ['No signs of pneumonia', 'Pneumonia detected']
    values = [total_images - pneumonia_detected, pneumonia_detected]
    
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#1f77b4', '#ff7f0e'])
    ax.axis('equal')
    st.pyplot(fig)
    
  # Ajoutez cette ligne en haut de votre fichier

# ...

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
            # Décoder les données si elles sont de type bytes
            image_name = row[1].decode('utf-8') if isinstance(row[1], bytes) else row[1]
            result = row[2].decode('utf-8') if isinstance(row[2], bytes) else row[2]
            
            # Traiter la probabilité
            if isinstance(row[3], bytes):
                # Si la probabilité est stockée sous forme de bytes, la convertir en float
                probability = struct.unpack('f', row[3])[0]  # Convertir les bytes en float
            else:
                # Sinon, convertir directement en float
                probability = float(row[3])
            
            st.write(f"Image : {image_name} | Result : {result} | probability : {probability:.2f}")
    else:
        st.write("No results recorded.")