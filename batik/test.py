import os
import streamlit as st
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt # type: ignore
import gdown

# CSS Styling untuk aplikasi Streamlit
st.markdown(
    f"""
    <style>
        .stApp {{
            background-color: #1F2833;
            color: #C5C6C7;
            font-family: 'Helvetica', sans-serif;
            padding: 20px;
        }}
        h1 {{
            color: #66FCF1;
            font-size: 2.5rem;
            text-align: center;
            font-weight: bold;
            margin-bottom: 1.5rem;
        }}
        h2 {{
            color: #45A29E;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stButton > button {{
            background-color: #45A29E;
            border: none;
            color: #0B0C10;
            padding: 0.5rem 1.5rem;
            font-size: 1rem;
            border-radius: 5px;
            margin: 10px 0;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }}
        .stButton > button:hover {{
            background-color: #66FCF1;
            color: #1F2833;
        }}
        .stImage > img {{
            border: 2px solid #45A29E;
            border-radius: 10px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Fungsi untuk memuat model Keras
@st.cache_resource
def load_keras_model(model_path):
    try:
        if not os.path.exists(model_path):
            st.error(f"File model tidak ditemukan di lokasi: {model_path}")
            return None
        model = load_model(model_path)
        st.success("✅ Model berhasil dimuat")
        return model
    except Exception as e:
        st.error(f"Model Error: {e}")
        return None

# Kelas label untuk motif batik
CLASS_NAMES = ['Motif Batik Aceh', 'Motif Batik Bali', 'Motif Batik Betawi', 'Motif Batik Cirebon', 'Motif Batik Lasem']

# Gambar contoh dengan path relatif
example_images = {
    "Motif Batik A": r"batik/sample_images/Aceh.jpg",
    "Motif Batik B": r"batik/sample_images/Bali.jpg",
    "Motif Batik C": r"batik/sample_images/Betawi.jpg",
    "Motif Batik D": r"batik/sample_images/Cirebon.jpg",
    "Motif Batik E": r"batik/sample_images/Lasem.jpg"
}

# Link file model Google Drive
vgg_model_url = "https://drive.google.com/uc?export=download&id=1wdrgADWhWdLIWjJ91pGMulhJJ8tyjH8L"
resnet_model_url = "https://drive.google.com/uc?export=download&id=1pJxyqm-LUsrq8B9iZIkRieAs2Ktmhrev"

# Path dimana model akan disimpan
vgg_model_path = "VGGModel3.h5"
resnet_model_path = "resnet50model3.h5"

gdown.download(vgg_model_url, vgg_model_path, quiet=False)
gdown.download(resnet_model_url, resnet_model_path, quiet=False)


# Fungsi untuk memproses gambar
def preprocess_image(image, target_size=(224, 224)):
    image = image.convert('RGB')
    image = image.resize(target_size) 
    img_array = img_to_array(image) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

# Fungsi untuk menampilkan grafik batang
def show_prediction_graph(predictions):
    prob_dict = dict(zip(CLASS_NAMES, predictions[0]))

    # Membuat grafik batang
    plt.figure(figsize=(10, 6))
    plt.barh(list(prob_dict.keys()), list(prob_dict.values()), color='#45A29E')
    plt.xlabel('Probabilitas', color='#C5C6C7')
    plt.title('Probabilitas untuk Setiap Kelas', color='#C5C6C7')
    plt.gca().invert_yaxis()  # Membalikkan urutan grafik
    plt.grid(color='#C5C6C7', linestyle='--', linewidth=0.5, axis='x')

    # Menampilkan grafik di Streamlit
    st.pyplot(plt)

# Update fungsi prediksi untuk menampilkan grafik
def predict_image(image, model, threshold=0.3):
    try:
        img_array = preprocess_image(image)
        predictions = model.predict(img_array)
        st.write(f"Predictions: {predictions}")  # Menampilkan probabilitas untuk setiap kelas

        if len(predictions) > 0:
            class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])

            # Menampilkan grafik probabilitas
            show_prediction_graph(predictions)

            if confidence < threshold:
                return "Motif Tidak Dikenali", confidence
            return CLASS_NAMES[class_idx], confidence
        else:
            st.error("Prediksi gagal, tidak ada hasil yang diterima dari model.")
            return None, None
    except Exception as e:
        st.error(f"Error saat melakukan prediksi: {e}")
        return None, None

# Header aplikasi
st.image(
    "https://youngontop.com/wp-content/uploads/2024/09/cara-membuat-batik.jpg",  
    use_column_width=True,
)
st.title("Klasifikasi Motif Batik")
st.write(
    "Aplikasi ini mengklasifikasikan gambar motif batik berdasarkan model yang telah dilatih. Unggah gambar untuk memulai prediksi."
)

st.markdown("### 📸 Contoh Klasifikasi Motif Batik")
with st.expander("Klik untuk melihat contoh gambar"):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(example_images["Motif Batik A"], caption="Motif Batik Aceh", use_column_width=True)
    with col2:
        st.image(example_images["Motif Batik B"], caption="Motif Batik Bali", use_column_width=True)
    with col3:
        st.image(example_images["Motif Batik C"], caption="Motif Batik Betawi", use_column_width=True)
    with col4:
        st.image(example_images["Motif Batik D"], caption="Motif Batik Cirebon", use_column_width=True)
    with col5:
        st.image(example_images["Motif Batik E"], caption="Motif Batik Lasem", use_column_width=True)

# Memilih model untuk digunakan
model_options = {
    "VGG Model": vgg_model_path,
    "ResNet Model": resnet_model_path
}
selected_model = st.selectbox("Pilih Model untuk Klasifikasi", options=list(model_options.keys()))

# Memuat model berdasarkan pilihan
model_path = model_options[selected_model]
model = load_keras_model(model_path)

# Unggah gambar
st.markdown("### 📂 Unggah Gambar Motif Batik Anda")
uploaded_file = st.file_uploader("Pilih gambar motif batik Anda (format: JPG, JPEG, atau PNG)", type=["jpg", "jpeg", "png"])

# Proses gambar unggahan
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    
    with st.spinner("⏳ Melakukan prediksi..."):
        label, confidence = predict_image(image, model, threshold=0.7)
    
    st.markdown("### 🔍 Hasil Prediksi")
    if label == "Motif Tidak Dikenali":
        st.error("⚠ Gambar motif batik tidak dapat dikenali.")
        st.info(f"*Tingkat Keyakinan:* {confidence:.2f}")
    elif label and confidence:
        st.success(f"✅ *Prediksi:* {label}")
        st.info(f"*Tingkat Keyakinan:* {confidence:.2f}")
    else:
        st.warning("⚠ Prediksi gagal dilakukan. Silakan coba lagi.")

# Footer
st.markdown("---")
st.write("📚 Aplikasi Klasifikasi Motif Batik")
