import streamlit as st
from cryptography.fernet import Fernet
from sklearn.ensemble import IsolationForest
import numpy as np
import os


# Set page config
st.set_page_config(page_title="Secure Two-Way Data Link", page_icon="🛩️")

key_path = os.path.join(os.getcwd(), "pages", "secret.key")
key = open(key_path, "rb").read()

# Encrypt Message
def encrypt_message(message, key):
    cipher = Fernet(key)
    return cipher.encrypt(message.encode())

# Decrypt Message
def decrypt_message(encrypted_message, key):
    cipher = Fernet(key)
    return cipher.decrypt(encrypted_message).decode()

# Train Isolation Forest Model
data = np.random.rand(100, 2)
anomaly = np.array([[0.9, 0.9]])
data = np.vstack([data, anomaly])

model = IsolationForest(contamination=0.05)
model.fit(data)

def detect_anomaly(input_data):
    prediction = model.predict([input_data])
    return "Anomaly Detected!" if prediction[0] == -1 else "Normal Data"


st.sidebar.title("S.O.P.H.I.E. Module - IV")
st.sidebar.info(
        "This is the fourth module of Series One Processor Hyper Intelligence Encryptor (S.O.P.H.I.E.). Please use it wisely")
st.sidebar.markdown("---")
st.sidebar.markdown("### General Instructions:")
st.sidebar.markdown("1. Enter message to encrypt if you want to do Encryption, click 'Encrypt & Send' ")
st.sidebar.markdown("2. Enter encrypted message: if you want to do Decryption, click 'Decrypt'' ")
st.sidebar.markdown("3. If you want to detect anomaly in data transmission, Enter the X & Y -coordinate, click 'Check Security'' ")
st.sidebar.markdown("---")


# Streamlit UI
st.title("🔐 Secure Two-Way Data Link")

# Encryption Section
message = st.text_input("Enter message to encrypt:")
if st.button("Encrypt & Send"):
    encrypted_msg = encrypt_message(message, key)
    st.success(f"Encrypted Message: {encrypted_msg}")

# Decryption Section
encrypted_input = st.text_input("Enter encrypted message:")
if st.button("Decrypt"):
    try:
        decrypted_msg = decrypt_message(encrypted_input.encode(), key)
        st.success(f"Decrypted Message: {decrypted_msg}")
    except:
        st.error("Invalid encrypted message!")

# ML-Based Security Check
st.subheader("📡 Anomaly Detection in Data Transmission")
x = st.number_input("Enter X-coordinate:")
y = st.number_input("Enter Y-coordinate:")
if st.button("Check Security"):
    result = detect_anomaly([x, y])
    st.warning(result)

 # Footer
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