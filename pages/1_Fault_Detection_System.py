import streamlit as st
import pandas as pd
import numpy as np
import random
from tensorflow.keras.models import load_model
from sklearn import preprocessing
from PIL import Image
import os

# Set page configuration
st.set_page_config(page_title="Aircraft Engine RUL Prediction System", page_icon="🛩️")

# Load the trained model
model_path = os.path.join(os.getcwd(), "pages", "rnn_model.h5")
model = load_model(model_path)

# Generate a random index
random_index = random.randint(0, 6)

# List of engine images
engine_images = [
    os.path.join("pages", "engine", "engine-1.jpg"),
    os.path.join("pages", "engine", "engine-2.jpg"),
    os.path.join("pages", "engine", "engine-3.jpg"),
    os.path.join("pages", "engine", "engine-4.jpg"),
    os.path.join("pages", "engine", "engine-5.jpg")
]
countries = ["USA", "USA", "European Union", "France", "Russia", "Russia", "China"]
engines = ["General Electric F110", "Pratt & Whitney F119", "Eurojet EJ200", "Snecma M88", "Klimov RD-33", "Lyulka AL-31F", "Shenyang WS-10"]
dry_trust = ["80 KN", "65 KN", "78 KN", "45 KN", "45 KN", "75 KN", "52 KN"]
wet_trust = ["165 KN", "125 KN", "110 KN", "75 KN", "65 KN", "105 KN", "95 KN"]
temp = os.path.join(os.getcwd(), "pages\\types\\")
image_paths = [temp + f"image-{i}.jpg" for i in range(7)]


# Sidebar
st.sidebar.title("S.O.P.H.I.E. Module - I")
st.sidebar.info("The is the first module of Series One Processor Hyper Intelligence Encryptor (S.O.P.H.I.E.). Please use it wisely")
st.sidebar.markdown("---")
st.sidebar.markdown("### General Instructions:")
st.sidebar.markdown("1. Enter test data in the text area.")
st.sidebar.markdown("2. Click the 'Predict' button.")
st.sidebar.markdown("3. View remaining life and next overhaul cycles of the Engine.")
st.sidebar.markdown("---")

# Load and preprocess input data
def preprocess_input_data(input_text):
    try:
        st.write("Raw input text received:")
        data = [list(map(float, line.split())) for line in input_text.strip().split('\n')]
        test_df = pd.DataFrame(data)
        test_df.iloc[:, :26]

        st.write("Initial DataFrame Shape:", test_df.shape)  # Debugging step

        if test_df.shape[1] != 26:
            st.error("Incorrect number of columns. Expected 26 columns.")
            return None

        cols_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
        test_df.columns = cols_names

        norm_cols = [col for col in test_df.columns if col not in ['id', 'cycle']]
        scaler = preprocessing.MinMaxScaler()
        test_df[norm_cols] = scaler.fit_transform(test_df[norm_cols])

        return test_df
    except Exception as e:
        st.error(f"Invalid input format: {e}")
        return None

def predict_rul(test_df, seq_length=50):
    seq_cols = ['s2']  # Ensure this feature exists in test_df
    test_sequences = [
        test_df[test_df['id'] == id][seq_cols].values[-seq_length:]
        for id in test_df['id'].unique()
        if len(test_df[test_df['id'] == id]) >= seq_length
    ]

    st.write(f"Generated sequences count: {len(test_sequences)}")  # Debugging step

    if not test_sequences:
        st.error("No valid test sequences found. Ensure at least 50 cycles per engine.")
        return []

    test_sequences = np.array(test_sequences, dtype=np.float32)

    return model.predict(test_sequences) if len(test_sequences) > 0 else []

# Streamlit UI
st.title("Aircraft Engine RUL Prediction System")

# Text area for input
test_data_input = st.text_area("Enter test data (space-separated values per row):")

if st.button("Predict Remaining Useful Life (RUL)"):
    test_data = preprocess_input_data(test_data_input)
    if test_data is not None:
        predictions = predict_rul(test_data)

        if not predictions:
            st.error("No valid predictions available. Ensure input data is correct.")
        else:
            for idx, pred in enumerate(predictions):
                remaining_life = int(pred[0] * 100)  # Scale appropriately
                overhaul_time = (remaining_life) / 3  # Example logic for next overhaul
                st.write(
                    f"Engine {idx + 1}: Estimated remaining life: {remaining_life} months. Next overhaul in {overhaul_time} months.")
        st.sidebar.markdown("### General Specification about the Engine:")
        st.sidebar.write(f"**Country:** {countries[random_index]}")
        st.sidebar.write(f"**Engine:** {engines[random_index]}")
        st.sidebar.write(f"**Dry Trust:** {dry_trust[random_index]}")
        st.sidebar.write(f"**Wet Trust:** {wet_trust[random_index]}")
        st.sidebar.image(image_paths[random_index], caption=f"Engine Image", use_container_width=True)

        # Randomly select an engine image to display
        random_image = random.choice(engine_images)
        image = Image.open(random_image)
        st.image(image, caption="Aircraft Engine", use_container_width=True)

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