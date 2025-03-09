import numpy as np
from PIL import Image
import tensorflow as tf
import streamlit as st
import requests
import tempfile
import os

MODEL_URL = "https://github.com/Imaaneea/cat_dog_classification_images/raw/master/cats_and_dogs_model.tflite"

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

# Fonction pour effectuer la prédiction sur l'image
def predict_image(image_file, model):
    try:
        # Charger et pré-traiter l'image
        image = Image.open(image_file)
        image = image.resize((224, 224))  # Redimensionner l'image à 224x224
        image = np.array(image)  # Convertir l'image en tableau numpy
        image = np.expand_dims(image, axis=0)  # Ajouter une dimension batch
        image = image / 255.0  # Normaliser l'image (assurez-vous que c'est le bon pré-traitement)

        # Préparer les entrées et obtenir la sortie
        input_details = model.get_input_details()
        output_details = model.get_output_details()

        # Configurer l'entrée du modèle
        model.set_tensor(input_details[0]['index'], image.astype(np.float32))
        model.invoke()  # Faire la prédiction

        # Obtenir les résultats
        output_data = model.get_tensor(output_details[0]['index'])
        class_index = np.argmax(output_data)  # Trouver la classe la plus probable
        confidence = np.max(output_data)  # La probabilité de la prédiction

        # Retourner le résultat sous forme de dictionnaire
        result = {
            'value': 'Cat' if class_index == 0 else 'Dog',  # 0: Cat, 1: Dog (selon ton modèle)
            'prob': confidence
        }

        return result
    except Exception as e:
        st.error(f"⛔ Erreur lors de la prédiction : {str(e)}")
        return {'value': 'Erreur', 'prob': 0}

# Charger le modèle
model = load_model()

# Streamlit Web App:
st.write("""
# MSDE5 : Deep Learning Project
## Cat Vs Dog Classification
""")

st.sidebar.image("https://miro.medium.com/v2/resize:fit:1400/format:webp/1*EvMbMNRHm_aOf1n4tDO1Xg.jpeg", width=250)
st.sidebar.write("This is a classification model of cat and dog images")
st.markdown("This project was made by : **EL AYACHI IMANE**")
st.write("Upload an image to classify whether it's a cat or a dog.")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Redimensionner l'image pour correspondre à la taille attendue par le modèle
    image = image.resize((224, 224))  # Résolution attendue : 224x224
    st.image(image, caption="Uploaded Image", use_container_width=True)

    result = predict_image(uploaded_file, model)
    
    # Afficher les résultats
    st.success(f"Prediction: It's a {result['value']}")
    st.success(f"Prob: {result['prob']}")

