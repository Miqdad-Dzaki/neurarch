import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Judul aplikasi
st.title("🧱 Deteksi Kerusakan Dinding (YOLOv8)")

# Load model
model = YOLO("best.pt")  # ganti dengan path ke best.pt

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar dinding", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Simpan gambar sementara
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()

    # Buka gambar
    image = Image.open(temp_file.name)
    st.image(image, caption="Gambar Input", use_column_width=True)

    # Prediksi dengan YOLO
    results = model.predict(source=temp_file.name, save=False, conf=0.25)

    # Tampilkan hasil deteksi
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])  # ID kelas
            conf = float(box.conf[0]) # confidence
            label = r.names[cls_id]   # nama kelas

            # Tampilkan deteksi
            st.write(f"✅ Deteksi: **{label}** (confidence: {conf:.2f})")

            # Berikan saran sesuai deteksi
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

    # Simpan hasil prediksi dengan bounding box
    results[0].save("output.jpg")
    st.image("output.jpg", caption="Hasil Deteksi", use_column_width=True)
