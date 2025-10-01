import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
import os

# Judul aplikasi
st.set_page_config(page_title="Deteksi Kerusakan Dinding", layout="wide")
st.title("🧱 Deteksi Kerusakan Dinding (YOLOv8)")

# Load model (pastikan file best.pt ada di folder yang sama dengan app.py)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# Upload file
uploaded_file = st.file_uploader("📂 Upload gambar atau video dinding", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    # Simpan sementara
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    file_type = uploaded_file.type

    # Jika input gambar
    if "image" in file_type:
        image = Image.open(temp_path)
        st.image(image, caption="🖼️ Gambar Input", use_column_width=True)

        results = model.predict(source=temp_path, save=False, conf=0.25)

        # Tampilkan deteksi
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = r.names[cls_id]

                st.write(f"✅ Deteksi: **{label}** (confidence: {conf:.2f})")

                if label == "wall_crack":
                    st.info("⚠️ Retak terdeteksi. Segera lakukan perbaikan untuk mencegah kerusakan lebih lanjut.")
                elif label == "wall_mold":
                    st.info("⚠️ Jamur terdeteksi. Periksa kelembaban ruangan dan lakukan pembersihan.")
                elif label == "wall_corrosion":
                    st.info("⚠️ Korosi terdeteksi. Segera lakukan perawatan pada permukaan yang rusak.")
                elif label == "wall_deterioration":
                    st.info("⚠️ Deteriorasi terdeteksi. Pertimbangkan renovasi pada area ini.")
                elif label == "wall_stain":
                    st.info("⚠️ Noda terdeteksi. Periksa sumber kelembaban atau kebocoran.")

        # Hasil deteksi dengan bounding box
        result_img = results[0].plot()
        st.image(result_img, caption="📍 Hasil Deteksi", use_column_width=True)

    # Jika input video
    elif "video" in file_type:
        st.video(temp_path)  # tampilkan video asli

        st.write("⏳ Sedang memproses video, harap tunggu...")

        cap = cv2.VideoCapture(temp_path)
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, save=False, conf=0.25, verbose=False)
            annotated_frame = results[0].plot()
            out.write(annotated_frame)

        cap.release()
        out.release()

        st.success("✅ Video selesai diproses")
        st.video(output_path)
