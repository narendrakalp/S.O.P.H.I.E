import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from PIL import Image

import os
from tensorflow.keras.models import load_model


def load_trained_model():
    model_path = os.path.join(os.getcwd(), "pages", "trained_model.h5")  # Ensure correct path
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        model = load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")


def get_data_generator(uploaded_files, img_size=(224, 224)):
    filepaths = []
    for file in uploaded_files:
        path = f"dataset{file.name}"
        filepaths.append(path)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
    return filepaths

# Set page configuration
st.set_page_config(page_title="Military Aircraft Detection System", page_icon="🛩️")

# Sidebar
st.sidebar.title("S.O.P.H.I.E. Module - III")
st.sidebar.info("The is the third module of Series One Processor Hyper Intelligence Encryptor (S.O.P.H.I.E.). Please use it wisely")
st.sidebar.markdown("---")
st.sidebar.markdown("### General Instructions:")
st.sidebar.markdown("1. Browse the image of Aircraft.")
st.sidebar.markdown("2. Click the 'Classify' button.")
st.sidebar.markdown("3. View the entered image along with type of Aircraft it is.")
st.sidebar.markdown("---")

# Streamlit UI
st.title("Military Aircraft Detection System")

uploaded_files = st.file_uploader("Upload Image(s)", type=["jpg", "png"], accept_multiple_files=True)

# Basic Type Information
image_paths = [f"pages/aircraft-types/image-{i}.jpg" for i in range(41)]
aircraft_data = [
    # Format: ("Aircraft", "Country", "Engine Used", "Max Takeoff Weight", "Current Status")
    ("A-10 Thunderbolt II", "USA", "General Electric TF34-GE-100", "22,700 kg", "Active"),
    ("A400M Atlas", "European Union", "EuroProp TP400-D6", "141,000 kg", "Active"),
    ("AG600 Kunlong", "China", "Dongan WJ-6", "53,500 kg", "Active"),
    ("AV-8B Harrier II", "USA", "Rolls-Royce Pegasus", "14,100 kg", "Active"),
    ("B-1 Lancer", "USA", "General Electric F101-GE-102", "216,400 kg", "Active"),
    ("B-2 Spirit", "USA", "General Electric F118-GE-100", "170,600 kg", "Active"),
    ("B-52 Stratofortress", "USA", "Pratt & Whitney TF33-P-3/103", "220,000 kg", "Active"),
    ("Be-200", "Russia", "Progress D-436TP", "37,200 kg", "Active"),
    ("C-130 Hercules", "USA", "Allison T56", "70,300 kg", "Active"),
    ("C-17 Globemaster III", "USA", "Pratt & Whitney F117-PW-100", "265,350 kg", "Active"),
    ("C-5 Galaxy", "USA", "General Electric TF39/CF6", "381,000 kg", "Active"),
    ("E-2 Hawkeye", "USA", "Allison T56-A-427A", "23,000 kg", "Active"),
    ("Eurofighter Typhoon", "European Union", "Eurojet EJ200", "23,500 kg", "Active"),
    ("F-117 Nighthawk", "USA", "General Electric F404-F1D2", "23,800 kg", "Retired"),
    ("F-14 Tomcat", "USA", "Pratt & Whitney TF30", "33,700 kg", "Retired"),
    ("F-15 Eagle", "USA", "Pratt & Whitney F100/General Electric F110", "36,700 kg", "Active"),
    ("F-16 Fighting Falcon", "USA", "General Electric F110/Pratt & Whitney F100", "19,200 kg", "Active"),
    ("F/A-18 Hornet", "USA", "General Electric F404/F414", "23,500 kg", "Active"),
    ("F-22 Raptor", "USA", "Pratt & Whitney F119-PW-100", "38,000 kg", "Active"),
    ("F-35 Lightning II", "USA", "Pratt & Whitney F135", "31,800 kg", "Active"),
    ("F-4 Phantom II", "USA", "General Electric J79", "28,000 kg", "Retired"),
    ("J-20", "China", "Shenyang WS-10", "39,600 kg", "Active"),
    ("JAS 39 Gripen", "Sweden", "Volvo RM12 (based on General Electric F404)", "14,000 kg", "Active"),
    ("MQ-9 Reaper", "USA", "Honeywell TPE331", "4,760 kg", "Active"),
    ("MiG-31", "Russia", "Soloviev D-30F6", "46,200 kg", "Active"),
    ("Mirage 2000", "France", "Snecma M53-P2", "17,000 kg", "Active"),
    ("RQ-4 Global Hawk", "USA", "Rolls-Royce AE 3007H", "14,600 kg", "Active"),
    ("Rafale", "France", "Snecma M88", "24,500 kg", "Active"),
    ("SR-71 Blackbird", "USA", "Pratt & Whitney J58", "78,000 kg", "Retired"),
    ("Su-34", "Russia", "Lyulka AL-31F", "45,000 kg", "Active"),
    ("Su-57", "Russia", "Saturn AL-41F1", "35,000 kg", "Active"),
    ("Tornado", "European Union", "Turbo-Union RB199", "28,000 kg", "Active"),
    ("Tu-160", "Russia", "Kuznetsov NK-32", "275,000 kg", "Active"),
    ("Tu-95", "Russia", "Kuznetsov NK-12", "188,000 kg", "Active"),
    ("U-2 Dragon Lady", "USA", "General Electric F118-GE-101", "18,600 kg", "Active"),
    ("US-2", "Japan", "Rolls-Royce AE 2100J", "47,700 kg", "Active"),
    ("V-22 Osprey", "USA", "Rolls-Royce AE 1107C", "27,400 kg", "Active"),
    ("Avro Vulcan", "United Kingdom", "Rolls-Royce Olympus", "204,000 kg", "Retired"),
    ("XB-70 Valkyrie", "USA", "General Electric YJ93", "250,000 kg", "Cancelled"),
    ("YF-23", "USA", "Pratt & Whitney YF119/General Electric YF120", "29,000 kg", "Cancelled")
]


if st.button("Classify (Predict the type of Aircraft)"):
    if uploaded_files:
        model = load_trained_model()
        img_paths = get_data_generator(uploaded_files)
        data_gen = ImageDataGenerator(rescale=1./255)
        images = np.array([np.array(Image.open(img).resize((224, 224))) for img in img_paths]) / 255.0
        predictions = model.predict(images)
        for img_path, pred in zip(img_paths, predictions):

            # Get the index of the predicted class
            predicted_index = np.argmax(pred) + 1

            # Extract the corresponding data from the aircraft_data list
            aircraft_info = aircraft_data[predicted_index]  # Tuple containing aircraft details

            # Unpack the tuple
            aircraft_name, country, engine_used, max_takeoff_weight, current_status = aircraft_info

            st.image(img_path, caption=f"Prediction: {aircraft_name}")
            st.sidebar.markdown("### General Specification about the Engine:")

            # Display aircraft details
            st.sidebar.write(f"**Aircraft Name:** {aircraft_name}")
            st.sidebar.write(f"**Country:** {country}")
            st.sidebar.write(f"**Engine Used:** {engine_used}")
            st.sidebar.write(f"**Max Takeoff Weight:** {max_takeoff_weight}")
            st.sidebar.write(f"**Current Status:** {current_status}")

            # Display the corresponding image (assuming `image_paths` is correctly mapped to `aircraft_data` indices)
            st.sidebar.image(image_paths[predicted_index + 1], caption=f"{aircraft_name}", use_container_width=True)


    else:
        st.error("Please upload images for classification.")
st.sidebar.markdown(
        """
        <style>
            .full-width-button img {
                width: 100% !important;
            }
        </style>
        <a href="https://amalprasadtrivediportfolio.vercel.app/" target="_blank" class="full-width-button">
            <img src="https://img.shields.io/badge/Created%20by-Amal%20Prasad%20Trivedi-blue">
        </a>
        """,
        unsafe_allow_html=True
)