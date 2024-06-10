import streamlit as st
import pandas as pd
import pickle
from utils import preprocess, clean
import os

@st.cache_resource
def load_model():
    with open('ecg_model/trained_model.pkl', 'rb') as f:
        model, scaler = pickle.load(f)
    return model, scaler

def preprocess_input(data, scaler):
    # Ensure the data has the right columns
    if data.shape[1] != 32:
        st.error("Input data must have 32 columns corresponding to the features used in training.")
        return None

    # Scale the data
    scaled_data = scaler.transform(data)
    return scaled_data

def main():
    st.set_page_config(page_title="ECG Classification App", page_icon="❤️", layout="centered")

    st.title("ECG Classification App")
    st.markdown("""
    Welcome to the ECG Classification App. This tool allows you to upload an ECG dataset in CSV format and get a classification prediction indicating whether the ECG is **Normal** or **Abnormal (Arrhythmia)**.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file containing ECG data", type="csv")

    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        st.write("## Uploaded Data")
        st.write(user_data)

        if st.button('Classify'):
            model, scaler = load_model()
            try:
                processed_data = preprocess_input(user_data, scaler)
                if processed_data is not None:
                    prediction = model.predict(processed_data)
                    prediction_label = 'Normal' if prediction[0] == 0 else 'Abnormal (Arrhythmia)'

                    st.write("## Prediction Result")
                    st.markdown(f"<h3 style='text-align: center; color: {'green' if prediction[0] == 0 else 'red'};'>{prediction_label}</h3>", unsafe_allow_html=True)

                    if prediction[0] == 0:
                        st.markdown("""
                        ### What does this mean?
                        **Normal**: The ECG data is classified as normal, indicating that the heart rhythm appears to be regular.
                        """)
                    else:
                        st.markdown("""
                        ### What does this mean?
                        **Abnormal (Arrhythmia)**: The ECG data is classified as abnormal, indicating that there may be irregularities in the heart rhythm.
                        - **Arrhythmia** refers to an irregular heart rhythm, which can be too fast, too slow, or erratic.
                        - It is important to consult a healthcare professional for a detailed assessment and diagnosis.
                        """)
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    st.markdown("""
    <style>
    .css-18e3th9 {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .css-1d391kg {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
