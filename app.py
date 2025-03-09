import streamlit as st
import requests
import tempfile
import os
import tensorflow as tf
import numpy as np
from PIL import Image


MODEL_URL = "https://github.com/Imaaneea/cat_dog_classification_images/raw/master/cats_and_dogs_model.tflite"

@st.cache_resource  # Cache pour éviter de télécharger à chaque exécution
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

    if os.path.exists(model_path):
        st.write(f"✅ Modèle téléchargé à : {model_path}")
        st.write(f"📂 Taille du fichier : {os.path.getsize(model_path)} octets")
    else:
        st.error("⛔ Le fichier du modèle n'a pas été téléchargé correctement.")
        return None

    try:
        model = tf.lite.Interpreter(model_path=model_path)
        model.allocate_tensors()
        st.success("✅ Modèle chargé avec succès !")
        return model
    except Exception as e:
        st.error(f"⛔ Erreur lors du chargement du modèle : {str(e)}")
        return None

def predict_image(image_file, model):
    """Prédit si l'image est un chat ou un chien."""
    try:
        # Charger et prétraiter l'image
        image = Image.open(image_file).convert("RGB")
        image = image.resize((150, 150))  
        image_array = np.array(image) / 255.0  
        image_array = np.expand_dims(image_array, axis=0).astype(np.float32)  

        # Récupérer les détails du modèle
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        # Vérifier si le format d'entrée du modèle est correct
        st.write(f"📌 Input Shape attendu : {input_details[0]['shape']}")
        st.write(f"📌 Image Shape fournie : {image_array.shape}")

        # Passer l'image au modèle
        model.set_tensor(input_details[0]['index'], image_array)
        model.invoke()

        # Récupérer la sortie
        prediction = model.get_tensor(output_details[0]['index'])[0][0]

        # Déterminer la classe
        class_names = ["Chat 🐱", "Chien 🐶"]
        result = {
            "value": class_names[int(prediction > 0.5)],
            "prob": round(float(prediction), 4),
        }
        return result

    except Exception as e:
        st.error(f"⛔ Erreur lors de la prédiction : {str(e)}")
        return {"value": "Erreur", "prob": 0}

# Charger le modèle
model = load_model()

# Interface Streamlit
st.write("""
# MSDE5 : Deep Learning Project
## Cat Vs Dog Classification
""")

st.sidebar.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*EvMbMNRHm_aOf1n4tDO1Xg.jpeg", width=250)
st.sidebar.write("This is a classification model of cat and dog images")
st.markdown("This project was made by : **KHAWLA BADDAR** & **Aymane ElAZHARI**")
st.write("Upload an image to classify whether it's a cat or a dog.")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    result = predict_image(uploaded_file, model)

    # Afficher le résultat
    st.success(f"🔍 Prédiction : {result['value']}")
    st.success(f"📊 Probabilité : {result['prob']}")
