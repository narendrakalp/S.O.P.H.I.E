import streamlit as st
from streamlit_option_menu import option_menu
import os


# Set page config
st.set_page_config(page_title="S.O.P.H.I.E.", page_icon="🛩️", layout="wide")

st.title("S.O.P.H.I.E. - (Series One Processor Hyper Intelligence Encryptor)")

# Sidebar
st.sidebar.title("S.O.P.H.I.E.")
st.sidebar.info(
        "This is the main module of Series One Processor Hyper Intelligence Encryptor (S.O.P.H.I.E.). Please use it wisely")
st.sidebar.markdown("---")
st.sidebar.markdown("### General Instructions:")
st.sidebar.markdown("1. This is the main module of S.O.P.H.I.E.")
st.sidebar.markdown("2. The software consist of 4 sub-modules.")
st.sidebar.markdown("3. Every module is design for specific purpose.")
st.sidebar.markdown("---")


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

# Title of the page
st.title("Welcome to S.O.P.H.I.E.")
st.write("""
    **S.O.P.H.I.E.** stands for **Series One Processor Hyper Intelligence Encryptor**, a cutting-edge software suite designed to address complex challenges in the fields of fault detection, target classification, military vehicle detection, and secure data transmission. This system combines AI-driven models with secure encrypted communication channels to ensure data integrity, security, and real-time anomaly detection.
""")

# About S.O.P.H.I.E.
st.markdown("""
    ## Features of S.O.P.H.I.E.

    ### 1. **Fault Detection System**
    The **Fault Detection System** is a powerful tool designed to identify abnormalities in mechanical and industrial systems. It leverages machine learning algorithms, specifically **Isolation Forests**, to detect faults in complex systems. By identifying deviations from normal behavior, this system helps prevent costly downtime, improve safety, and optimize operations.

    **How It Works**: This system is based on anomaly detection techniques, where the model is trained using data from operational systems. The model then learns to identify the patterns of normal operation and flags any data points that diverge from these patterns as potential faults. This capability makes it invaluable for industries like manufacturing, transportation, and energy.
    
    ### 2. **Micro-Doppler based Target Classification System**
    The **Micro-Doppler based Target Classification System** uses advanced radar signal processing techniques to classify moving objects based on their Doppler shifts. This system is ideal for military and surveillance applications where distinguishing between various types of moving targets (e.g., vehicles, drones) is critical.

    **How It Works**: By analyzing the Doppler shifts in radar signals, this system can classify targets based on their motion characteristics. The classification model is powered by machine learning algorithms that learn patterns associated with different types of targets. Once trained, it can identify and classify incoming objects in real-time, providing invaluable data for military and security operations.

    ### 3. **Military Vehicles Detection System**
    The **Military Vehicles Detection System** is designed to automatically detect and classify military vehicles in satellite imagery or video feeds. This is crucial for defense and security agencies to monitor and track enemy movements in real-time.

    **How It Works**: Using deep learning-based object detection models like **YOLO** or **Faster R-CNN**, the system analyzes visual data and detects military vehicles. The system is trained with large datasets of vehicle images to identify vehicles even under challenging conditions. It can process images quickly, making it suitable for real-time surveillance.

    ### 4. **Two-Way Signal Processing (Secure Two-Way Data Link)**
    The **Two-Way Signal Processing** system ensures secure, encrypted communication between two endpoints, making it essential in applications requiring data security and integrity, such as military communications and secure business transactions.

    **How It Works**: The system uses advanced encryption techniques to secure data transmission between two parties. Through symmetric encryption (AES) and public key infrastructure (RSA), data is encrypted and authenticated before transmission. The system ensures uninterrupted communication by handling network issues such as packet loss and jitter.

    ---
    ## S.O.P.H.I.E. as a Comprehensive Solution

    While each of these systems is powerful on its own, **S.O.P.H.I.E.** integrates them into one cohesive suite, allowing users to address multiple challenges in a unified manner.

    ### Key Benefits of Using S.O.P.H.I.E.:
    - **Versatility**: S.O.P.H.I.E. addresses challenges in various sectors including defense, industrial operations, and security.
    - **Real-Time Processing**: With AI models optimized for real-time processing, S.O.P.H.I.E. delivers immediate feedback on critical tasks.
    - **Advanced Security**: Built-in security ensures that all communications are encrypted, making it ideal for sensitive data.
    - **Machine Learning Integration**: Powered by machine learning, S.O.P.H.I.E. continuously improves performance and adapts to new data.
    - **Scalability**: The system is designed to scale with your needs, whether you're working on small projects or enterprise-level systems.
    - **Ease of Use**: The intuitive interface of **S.O.P.H.I.E.** makes it accessible for all users, whether you’re a data scientist or a business manager.

    ---
    ## Future Developments

    **S.O.P.H.I.E.** will continue to evolve. We are working on several exciting developments:

    - **Integration with IoT**: Connecting **S.O.P.H.I.E.** with IoT devices will enable real-time data collection, improving fault detection accuracy and other systems.
    - **More Machine Learning Models**: Upcoming versions will feature more advanced machine learning models for enhanced classification, anomaly detection, and predictive analytics.
    - **Cloud-based Deployment**: **S.O.P.H.I.E.** will soon be available on the cloud, allowing for flexible, scalable deployment with remote access.

    ---
    ## Conclusion

    **S.O.P.H.I.E.** is not just a tool—it’s a comprehensive, intelligent, and secure solution for solving complex problems in modern operations. Whether in defense, industry, or business, **S.O.P.H.I.E.** offers powerful AI-driven systems that guarantee enhanced performance, security, and real-time insights.

    With **S.O.P.H.I.E.**, you can ensure the safety of operations, secure communication, and intelligent decision-making. We invite you to explore **S.O.P.H.I.E.** and leverage its capabilities to enhance your processes and achieve greater operational efficiency.

    Thank you for choosing **S.O.P.H.I.E.**, your partner in smart, secure, and efficient solutions.
""")

# Style for the markdown text
st.markdown("""
    <style>
        .css-1d391kg { 
            font-size: 16px;
            line-height: 1.6;
            color: #333;
        }
        .css-1d391kg h1 {
            font-size: 32px;
            font-weight: bold;
            text-align: center;
        }
        .css-1d391kg h2 {
            font-size: 28px;
            color: #1f77b4;
        }
        .css-1d391kg h3 {
            font-size: 24px;
            color: #d62728;
        }
        .css-1d391kg p {
            text-align: justify;
            font-size: 16px;
        }
        .css-1d391kg li {
            font-size: 16px;
            margin-bottom: 10px;
        }
        .css-1d391kg ul {
            list-style-type: square;
            padding-left: 20px;
        }
    </style>
""", unsafe_allow_html=True)