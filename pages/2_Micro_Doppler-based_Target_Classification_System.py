import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import io
from PIL import Image

# Set page config
st.set_page_config(page_title="Micro-Doppler Target Classification System", page_icon="🛩️")

# Generate synthetic data
@st.cache_data
def generate_data(n_samples=10000):
    np.random.seed(42)

    # Generate features
    frequency = np.random.uniform(1, 10, n_samples)
    amplitude = np.random.uniform(0, 1, n_samples)
    duration = np.random.uniform(0.1, 2, n_samples)

    # Generate labels (0 for bird, 1 for drone)
    labels = np.random.choice([0, 1], n_samples)

    # Adjust features based on labels
    frequency += labels * np.random.uniform(0, 5, n_samples)  # Drones tend to have higher frequency
    amplitude += labels * np.random.uniform(0, 0.5, n_samples)  # Drones tend to have higher amplitude
    duration -= labels * np.random.uniform(0, 0.5, n_samples)  # Birds tend to have longer duration

    # Create DataFrame
    df = pd.DataFrame({
        'frequency': frequency,
        'amplitude': amplitude,
        'duration': duration,
        'label': labels
    })

    df['class'] = df['label'].map({0: 'Bird', 1: 'Drone'})
    df['timestamp'] = [datetime.now() - timedelta(minutes=i) for i in range(n_samples)]

    return df


# Load or generate data
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv('train_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    except FileNotFoundError:
        df = generate_data()
        df.to_csv('micro_doppler_data.csv', index=False)
    return df


# Train classification model
@st.cache_resource
def train_model(df):
    X = df[['frequency', 'amplitude', 'duration']]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return model, accuracy


# Main application
def main():
    st.title("Micro-Doppler Target Classification System")

    # Sidebar
    st.sidebar.title("S.O.P.H.I.E. Module - II")
    st.sidebar.info(
        "This is the second module of Series One Processor Hyper Intelligence Encryptor (S.O.P.H.I.E.). Please use it wisely")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### General Instructions:")
    st.sidebar.markdown("1. Upload test data in the csv format.")
    st.sidebar.markdown("2. Click the 'Detect New Target' button.")
    st.sidebar.markdown("3. View weather the entered data is of bird or drone.")
    st.sidebar.markdown("---")

    # Load data and train model
    df = load_data()
    model, model_accuracy = train_model(df)

    # Sidebar for date range selection
    st.sidebar.title("Date range of Data")
    date_range = st.sidebar.date_input(
        "The data of this trained model is taken from following duration.",
        value=(df['timestamp'].min().date(), df['timestamp'].max().date()),
        min_value=df['timestamp'].min().date(),
        max_value=df['timestamp'].max().date()
    )

    # Filter data based on date range
    mask = (df['timestamp'].dt.date >= date_range[0]) & (df['timestamp'].dt.date <= date_range[1])
    filtered_df = df.loc[mask]

    # Main content
    tab1, tab2, tab3= st.tabs(["Data Analysis", "Classification", "Real-time Detection"])

    with tab1:
        st.header("Data Analysis")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Observations", len(filtered_df))
        with col2:
            st.metric("Drones Detected", len(filtered_df[filtered_df['class'] == 'Drone']))
        with col3:
            st.metric("Birds Detected", len(filtered_df[filtered_df['class'] == 'Bird']))

        # Micro-Doppler signature visualization
        st.subheader("Micro-Doppler Signature Visualization")
        selected_class = st.selectbox("Select Target Class", ['All', 'Drone', 'Bird'])

        if selected_class != 'All':
            plot_df = filtered_df[filtered_df['class'] == selected_class]
        else:
            plot_df = filtered_df

        fig = px.scatter_3d(plot_df, x='frequency', y='amplitude', z='duration', color='class',
                            title="3D Micro-Doppler Signature Plot")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The 3D plot shows the distribution of micro-Doppler signatures for drones and birds.
        Drones tend to have higher frequency and amplitude, while birds often have longer duration signatures.
        </div>
        """, unsafe_allow_html=True)

        # Time series of detections
        st.subheader("Detection Timeline")
        timeline_df = filtered_df.groupby(['timestamp', 'class']).size().unstack(fill_value=0).reset_index()
        fig_timeline = px.line(timeline_df, x='timestamp', y=['Drone', 'Bird'],
                               title="Target Detections Over Time")
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The timeline shows the frequency of drone and bird detections over time.
        This can help identify patterns or unusual activities in the monitored airspace.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.header("Classification")

        # Model performance
        st.subheader("Model Performance")
        st.metric("Model Accuracy", f"{model_accuracy:.2f}")

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': ['frequency', 'amplitude', 'duration'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig_importance = px.bar(feature_importance, x='importance', y='feature', orientation='h',
                                title="Feature Importance for Target Classification")
        st.plotly_chart(fig_importance, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The feature importance plot shows which micro-Doppler characteristics
        are most crucial for distinguishing between drones and birds. This information can guide the
        development of more targeted detection strategies.
        </div>
        """, unsafe_allow_html=True)

        # Confusion Matrix
        y_true = df['label']
        y_pred = model.predict(df[['frequency', 'amplitude', 'duration']])
        cm = confusion_matrix(y_true, y_pred)

        fig_cm = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                           x=['Bird', 'Drone'], y=['Bird', 'Drone'],
                           title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("""
        <div class="insight-box">
        <strong>Insight:</strong> The confusion matrix provides a detailed view of the model's performance,
        showing how well it distinguishes between drones and birds. This helps in understanding any
        misclassifications and potential areas for improvement.
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.header("Real-time Detection Simulation")

        uploaded_files = st.file_uploader("Upload CSV file", type=["csv"], accept_multiple_files=False)

        if uploaded_files:
            try:
                df = pd.read_csv(uploaded_files)

                # Ensure required columns exist
                required_columns = {'frequency', 'amplitude', 'duration'}
                if not required_columns.issubset(df.columns):
                    st.error(f"Uploaded file must contain columns: {', '.join(required_columns)}")
                else:
                    st.subheader("Simulated Target Detection")

                    if st.button("Detect New Target"):
                        new_target = df.iloc[0]  # Selecting first row

                        # Extract features
                        features = new_target[['frequency', 'amplitude', 'duration']].values.reshape(1, -1)

                        # Prediction
                        prediction = model.predict(features)[0]
                        prediction_proba = model.predict_proba(features)[0]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Detected Class", "Drone" if prediction == 1 else "Bird")
                        with col2:
                            st.metric("Confidence", f"{max(prediction_proba):.2f}")

                        # Radar chart visualization
                        fig_radar = go.Figure(data=go.Scatterpolar(
                            r=[new_target['frequency'], new_target['amplitude'], new_target['duration']],
                            theta=['Frequency', 'Amplitude', 'Duration'],
                            fill='toself'
                        ))
                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                            showlegend=False
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

                        st.markdown("""
                        <div class="insight-box">
                        <strong>Insight:</strong> The radar chart visualizes the micro-Doppler features of the detected target.
                        This representation helps in quickly assessing the characteristics that led to the classification decision.
                        </div>
                        """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error processing file: {e}")

    # Footer
    st.sidebar.markdown("---")

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

if __name__ == "__main__":
    main()