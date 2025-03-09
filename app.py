import streamlit as st
import requests
import tempfile
import os
import tensorflow as tf

MODEL_URL = "https://github.com/Imaaneea/cat_dog_classification_images/blob/master/APP%20deploiement/cats_and_dogs_model.tflite"

@st.cache_resource  # Utilisation de st.cache_resource
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

    # Vérifier si le fichier existe et sa taille
    if os.path.exists(model_path):
        st.write(f"✅ Modèle téléchargé à : {model_path}")
        st.write(f"📂 Taille du fichier : {os.path.getsize(model_path)} octets")
    else:
        st.error("⛔ Le fichier du modèle n'a pas été téléchargé correctement.")
        return None

    # Charger le modèle TensorFlow Lite
    try:
        model = tf.lite.Interpreter(model_path=model_path)
        model.allocate_tensors()
        st.success("✅ Modèle chargé avec succès !")
        return model
    except Exception as e:
        st.error(f"⛔ Erreur lors du chargement du modèle : {str(e)}")
        return None

# Charger le modèle
model = load_model()
