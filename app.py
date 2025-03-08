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
        st.error("Failed to download the model.")
        return None

    model_content = response.content
    model = tf.lite.Interpreter(model_content=model_content)
    model.allocate_tensors()
    return model

# Fonction pour faire des prédictions
def predict_image(img_to_predict, model):
    img = Image.open(img_to_predict).convert('RGB')
    img = img.resize((200, 200))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    input_details = model.get_input_details()
    output_details = model.get_output_details()

    model.set_tensor(input_details[0]['index'], img_array)
    model.invoke()
    prediction = model.get_tensor(output_details[0]['index'])

    if prediction[0, 0] > 0.5:
        return {'value': 'Dog', 'prob': f"{prediction[0, 0]:.4f}"}
    else:
        return {'value': 'Cat', 'prob': f"{1 - prediction[0, 0]:.4f}"}

# Charger le modèle
model = load_model()

if model is not None:
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

        st.success(f"Prediction: It's a {result['value']}")
        st.success(f"Probability: {result['prob']}")

else:
    st.error("Model could not be loaded. Please check the URL or try again later.")
