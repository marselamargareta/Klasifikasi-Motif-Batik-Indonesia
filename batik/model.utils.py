import gdown
import os
from tensorflow.keras.models import load_model  # type: ignore

# Fungsi untuk mengunduh model dari Google Drive
def download_model_from_gdrive(model_url, model_path):
    # Mendapatkan ID file dari URL Google Drive
    file_id = model_url.split('/d/')[1].split('/')[0]
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Fungsi untuk memuat model Keras
def load_keras_model(model_url, model_path):
    try:
        # Cek apakah model sudah ada di direktori lokal
        if not os.path.exists(model_path):
            # Jika tidak ada, unduh model
            download_model_from_gdrive(model_url, model_path)

        # Memuat model dari file
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
