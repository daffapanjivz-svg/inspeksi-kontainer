import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Judul Aplikasi
st.title("AI Digital Inspection - Kontainer")
st.write("Unggah foto kontainer untuk mendeteksi kerusakan secara otomatis.")

# Load Model (Pastikan file bernama best.pt ada di GitHub)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Fitur Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar kontainer...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Konversi gambar ke format yang bisa dibaca AI
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)
    
    st.write("Sedang menganalisis...")
    
    # Proses Deteksi
    results = model(image)
    
    # Tampilkan Hasil
    res_plotted = results[0].plot()
    res_image = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
    
    st.image(res_image, caption="Hasil Deteksi Kerusakan", use_container_width=True)
    st.success("Analisis Selesai!")
