import streamlit as st
import requests
from PIL import Image
import numpy as np
import tensorflow as tf
from io import BytesIO

# Fonction pour charger le modèle
@st.cache_resource
def load_model():
    model_url = "https://raw.githubusercontent.com/Imaaneea/cat_dog_classification_images/master/cats_and_dogs_model.tflite"

    response = requests.get(model_url)
    
    if response.status_code != 200:
        st.error(f"Failed to download the model. Status code: {response.status_code}")
        return None

    model_content = response.content  # Bytes
    model_bytes = bytearray(model_content)  # Convertir en bytearray

    try:
        model = tf.lite.Interpreter(model_content=model_bytes)
        model.allocate_tensors()
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Charger le modèle
model = load_model()

if model is not None:
    st.write("""
    # Cat Vs Dog Classification
    """)

    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
