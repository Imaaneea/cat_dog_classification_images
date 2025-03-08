import streamlit as st
import requests
import tempfile
import tensorflow as tf

# URL du modèle stocké sur Google Drive (avec lien de téléchargement direct)
MODEL_URL = "https://drive.google.com/uc?export=download&id=1xSIDZ3sHrFTXbwbGXY1I9BSCvpYvKMOC"

@st.cache_resource  # Remplace `st.cache` (obsolète) par `st.cache_resource`
def load_model():
    """Télécharge et charge le modèle TFLite"""
    with st.spinner("Téléchargement du modèle..."):
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tflite") as temp_model_file:
                for chunk in response.iter_content(chunk_size=1024):
                    temp_model_file.write(chunk)
                model_path = temp_model_file.name
        else:
            st.error("Erreur lors du téléchargement du modèle.")
            return None

    # Charger le modèle TensorFlow Lite
    model = tf.lite.Interpreter(model_path=model_path)
    model.allocate_tensors()
    return model

# Charger le modèle au démarrage de l'application
model = load_model()
if model:
    st.success("Modèle chargé avec succès ! 🎉")
