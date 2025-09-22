
import pandas as pd
import streamlit as st
import numpy as np 
import mysql.connector
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow import keras
import os
from tensorflow.keras.applications.resnet50 import preprocess_input
from ultralytics import YOLO

model_yolo = YOLO("my_model.pt")

model1 = keras.models.load_model("resnet.keras")
class_names=['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']
def predict_single_image(model, img_path, class_names, target_size=(224,224)):
    # Load image
    img = image.load_img(img_path, target_size=target_size)
    x = image.img_to_array(img)        # Convert to numpy array
    x = np.expand_dims(x, axis=0)      # Add batch dimension
    x = preprocess_input(x)            # ResNet50 preprocessing ([-1,1] scaling)

    # Predict
    preds = model.predict(x)
    y_pred = np.argmax(preds,axis=1)

    pred_class = class_names[y_pred[0]]  # Get class label
    confidence = np.max(preds)
    return pred_class, confidence

st.set_page_config(page_title="solar panel detection Dashboard", layout="wide")
st.title("☀️🛰️⚡ solar panel detection")
st.markdown(
    """
    <style>
    .main {
        border: 5px solid #4CAF50;   /* Outline color */
        border-radius: 15px;         /* Rounded corners */
        padding: 200px;               /* Space inside */
        margin: 100px;                /* Space outside */
        background-color: #f9f9f9;   /* Optional background */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("""
    <style>
    /* Tabs container */
    .stTabs {
        display: flex;
        justify-content: center;
    }

    /* All tab styles */
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 6px;
        padding: 30px 80px;
        margin-right: 50px;
        font-size: 18px;
        font-weight: 600;              /* Bold */
        font-style: italic;            /* Italic */
        font-family: 'Segoe UI', sans-serif;  /* Font family */
        text-transform: uppercase;     /* Make text uppercase */
        color: #333333;
    }

    /* Selected tab style */
    .stTabs [aria-selected="true"] {
        background-color: #2c7be5;
        color: white;
        font-weight: 700;
        font-style: normal;
    }
    </style>
    """, unsafe_allow_html=True)
html_temp="""<div style="background-color:#706C26;padding:10px">
                <h2 style="color:white;text-align:center;">solar panel fault detection</h2>
                </div>
                """
st.markdown(html_temp,unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🏠 Home", "🔍 prediction","object Detection"])
with tab1:
    st.markdown(f"""<h2 style="color: #706C26; font-size: 36px;">Solar Panel Fault Detection </h2>
                        <p style="font-size:18px; font-family: 'Segoe UI', sans-serif;word-spacing: 15px font-style: italic;"> &nbsp;&nbsp;&nbsp;Solar panel fault detection is the process of identifying physical or electrical issues within a solar panel or system that reduce energy output and reliability. Common faults include surface impurities like dust and cracks, physical defects like scratches and gaps, and electrical problems such as open circuits or ground faults. Techniques used for detection range from thermal imaging with drones to electrical tests, and increasingly, machine learning algorithms applied to images and performance data are used for automated, precise, and efficient fault localization and classification.   
                        </p>
                        <p style="color: #706C26; font-size: 36px;">Key Components of a Faulty solar panels dataset:</p>
                            <p style="font-size:18px; font-family: 'Segoe UI', sans-serif;word-spacing: 15px font-style: italic;">
                            <li>🌫️Dusty Images showing accumulation of dust particles that may reduce solar efficiency.</li>
                            <li>🐦 Bird-Drop Images containing bird droppings that obscure the panel surface.</li>
                            <li>⚡ Electrical-Damage Images with visible electrical damage such as burn marks, wiring faults, or short circuits.</li>
                            <li>🧱 Physical-Damage Images capturing physical issues like cracks, broken glass, or dents.</li>
                            <li>❄️Snow-Covered Images depicting panels covered in snow, which blocks sunlight absorption. </li></p>
                            <p style="color: #706C26; font-size: 36px;">Detection Methods</p>
                            <p style="font-size:18px; font-family: 'Segoe UI', sans-serif;word-spacing: 15px font-style: italic;"> <li>Thermal Imaging: Drones equipped with infrared cameras can capture images of solar panels to identify "hot spots" caused by electrical or physical defects. </li>
                                        <li>Electrical Testing: Ground fault locators and other instruments can test the electrical circuits of solar panels to pinpoint specific issues like loose connections. </li>
                                        <li>Image Processing & Computer Vision: Advanced algorithms can analyze images to detect surface impurities, cracks, and other visible defects. </li>
                                        <li>Machine Learning: Deep learning models, such as Convolutional Neural Networks (CNNs), are used to automate the analysis of thermal or visual images, enabling precise fault identification and classification. </li>
                                        <li>Time-Series Analysis: By monitoring a panel's power output data over time, models like ARIMA can forecast expected energy generation, and deviations from the forecast can signal potential problems with the panel. </li>
                                            </p>"""
                            ,unsafe_allow_html=True)

with tab2:


    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        save_path = os.path.join("tempDir", uploaded_file.name)
        os.makedirs("tempDir", exist_ok=True)

        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    button1=st.button("Predict")
    if button1:
        predicted=predict_single_image(model1, save_path, class_names, target_size=(224,224))
        st.write(predicted[0])
        st.markdown(f"""<div style="background-color:#C00025; padding:20px; border-radius:10px;
                            border: 1px solid #ddd; box-shadow: 2px 2px 8px rgba(0,0,0,0.05);">
                            <h4 style="color:white;text-align:center;"><br> The Solar Panel is {predicted[0]}</h4>""",unsafe_allow_html=True)

with tab3:
    uploaded_file1 = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="uploader2")
    if uploaded_file1 is not None:
        save_path1 = os.path.join("tempDir", uploaded_file1.name)
        os.makedirs("tempDir", exist_ok=True)

        with open(save_path1, "wb") as f:
            f.write(uploaded_file1.getbuffer())
    button2=st.button("Predict",key="button2")
    if button2:
        results=model_yolo.predict(save_path1)
        annotated_img = results[0].plot()

        st.image(annotated_img, caption="Detected Objects",use_container_width=True)
        st.write("Detection Results:")
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"- {model_yolo.names[cls]}: {conf:.2f}")
