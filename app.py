# Import library
import cv2
import streamlit as st
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import sys
import os


# Konfigurasi direktori
FILE = Path(__file__).resolve()
ROOT = FILE.parent

if ROOT not in sys.path:
    sys.path.append(str(ROOT))

# Folder utama
IMAGES_DIR = ROOT / 'images'
VIDEOS_DIR = ROOT / 'videos'
MODEL_DIR = ROOT / 'weights'

# File default
DEFAULT_IMAGE = IMAGES_DIR / 'image2.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'detectedimage2.jpg'

# Ambil daftar model .pt di folder 'weights'
model_files = [f.name for f in MODEL_DIR.glob("*.pt")]

# Jika tidak ada file model ditemukan
if not model_files:
    st.error("Tidak ada file model .pt ditemukan di folder 'weights'.")
    st.stop()

# Dropdown pilihan model
selected_model_file = st.sidebar.selectbox("Pilih Model YOLOv11", model_files)

# Load model berdasarkan pilihan
DETECTION_MODEL = (MODEL_DIR / selected_model_file).resolve()


# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Hama Tanaman - YOLOv11",
    page_icon="🪰",
    layout="wide"
)
st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
    </style>
""", unsafe_allow_html=True)


# Header
st.header("Deteksi Hama Tanaman menggunakan YOLOv11 (Model Custom)")

# Sidebar - Konfigurasi model
st.sidebar.header("🛠️ Konfigurasi Model")

confidence_value = float(st.sidebar.slider(
    "Pilih Nilai Confidence (kepercayaan deteksi)", 25, 100, 40)) / 100

# Load model YOLO
try:
    model = YOLO(DETECTION_MODEL)
    st.sidebar.success("✅ Model berhasil dimuat")
except Exception as e:
    st.sidebar.error(f"❌ Gagal memuat model dari: {DETECTION_MODEL}")
    st.sidebar.error(e)

# Sidebar - Pilih jenis input
st.sidebar.header("🎞️ Pilih Sumber Deteksi")
input_type = st.sidebar.radio("Deteksi pada:", ["Gambar", "Video"])

# ============================
# === DETEKSI PADA GAMBAR ===
# ============================
if input_type == "Gambar":
    st.sidebar.header("🖼️ Upload Gambar")
    uploaded_file = st.sidebar.file_uploader(
        "Pilih gambar hama (.jpg/.png)", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Asli")
        if uploaded_file is None:
            image = Image.open(str(DEFAULT_IMAGE))
            st.image(image, caption="Contoh Gambar", use_container_width=True)
        else:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang Diunggah", use_container_width=True)

    with col2:
        st.subheader("Hasil Deteksi")
        if uploaded_file is None:
            st.image(DEFAULT_DETECT_IMAGE, caption="Contoh Hasil Deteksi", use_container_width=True)
        else:
            if st.sidebar.button("🔍 Deteksi Gambar"):
                with st.spinner("Model sedang memproses gambar..."):
                    result = model.predict(image, conf=confidence_value)
                    result_plot = result[0].plot()[:, :, ::-1]
                    st.image(result_plot, caption="Hasil Deteksi", use_container_width=True)
                    with st.expander("📋 Rincian Deteksi"):
                        for box in result[0].boxes:
                            st.write(box.data)

# === MODE: VIDEO ===
elif input_type == "Video":
    st.sidebar.header("🎥 Upload Video")
    uploaded_video = st.sidebar.file_uploader(
        "Pilih video hama dari komputer Anda", 
        type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_video is not None:
        st.subheader("🧠 Deteksi Objek pada Video")

        # Simpan file video sementara
        video_path = f"temp_{uploaded_video.name}"
        with open(video_path, 'wb') as f:
            f.write(uploaded_video.read())

        if st.sidebar.button("▶️ Jalankan Deteksi Video"):
            try:
                video_cap = cv2.VideoCapture(video_path)
                st_frame = st.empty()

                while video_cap.isOpened():
                    success, frame = video_cap.read()
                    if not success:
                        break

                    frame = cv2.resize(frame, (720, int(720 * (9/16))))
                    results = model.predict(frame, conf=confidence_value)
                    result_frame = results[0].plot()
                    st_frame.image(result_frame, channels="BGR", use_container_width=True)

                video_cap.release()

                if os.path.exists(video_path):
                    os.remove(video_path)

            except Exception as e:
                st.error(f"❌ Gagal membaca video: {e}")



st.markdown("""---""")
st.markdown("""
    <center>
    Dibuat dengan ❤️ oleh Muhammad Ferditiar Yusuf<br>
    Menggunakan YOLOv11 dan Streamlit 🌿
    </center>
""", unsafe_allow_html=True)
