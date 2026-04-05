import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1. Judul & Header Web
st.set_page_config(page_title="Insepksi Kontainer AI", layout="wide")
st.title("🏗️ AI Digital Inspection - Container Damage")
st.write("Sistem otomatis untuk mendeteksi Karat (Rust), Penyok (Dent), dan Lubang (Hole).")

# 2. Fungsi Memuat Model (Agar Cepat)
@st.cache_resource
def load_model():
    # Pastikan file best.pt ada di folder yang sama dengan file ini
    return YOLO("best.pt")

model = load_model()

# 3. Sidebar untuk Pengaturan
st.sidebar.header("Pengaturan AI")
confidence = st.sidebar.slider("Tingkat Kepercayaan (Confidence)", 0.0, 1.0, 0.4)

# 4. Area Upload Foto
uploaded_file = st.file_uploader("Upload Foto Kontainer Anda...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Membuka gambar
    image = Image.open(uploaded_file)
    
    # Membuat dua kolom (Kiri: Foto Asli, Kanan: Hasil AI)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Foto Asli")
        st.image(image, use_container_width=True)
    
    # Menjalankan Prediksi AI
    results = model.predict(source=image, conf=confidence)
    
    with col2:
        st.subheader("Hasil Deteksi AI")
        # Menggambar kotak deteksi pada gambar
        res_plotted = results[0].plot()
        st.image(res_plotted, use_container_width=True)

    # 5. Ringkasan Temuan (Laporan Digital)
    st.divider()
    st.subheader("📝 Laporan Temuan Lapangan")
    
    # Menghitung jumlah kerusakan
    labels = results[0].names
    counts = {}
    for c in results[0].boxes.cls:
        name = labels[int(c)]
        counts[name] = counts.get(name, 0) + 1
    
    if counts:
        # Menampilkan tabel ringkasan
        for label, jumlah in counts.items():
            st.warning(f"Terdeteksi **{jumlah} titik {label}** pada kontainer ini.")
        st.info("Saran: Segera lakukan perbaikan pada area yang ditandai merah.")
    else:
        st.success("✅ Kontainer dalam kondisi BAIK (Clean). Tidak ditemukan kerusakan.")
