import streamlit as st
import pandas as pd
import pickle
import requests

# call the API

# if uploaded_file:

#     url = 'https://heartbeat-decoder-7pl35comwq-uk.a.run.app/predict'


#     info = {_0preRR:'0_pre-RR',
# _0postRR:'0_post-RR',
# _0pPeak:'0_pPeak',
# _0tPeak:'0_tPeak',
# _0rPeak:'0_rPeak',
# _0sPeak:'0_sPeak',
# _0qPeak:'0_qPeak',
# _0qrsinterval:'0_qrs_interval',
# _0pqinterval:'0_pq_interval',
# _0qtinterval:'0_qt_interval',
# _0stinterval:'0_st_interval',
# _0qrsmorph0:'0_qrs_morph0',
# _0qrsmorph1:'0_qrs_morph1',
# _0qrsmorph2:'0_qrs_morph2',
# _0qrsmorph3:'0_qrs_morph3',
# _0qrsmorph4:'0_qrs_morph4',
# _1preRR:'1_pre-RR',
# _1postRR:'1_post-RR',
# _1pPeak:'1_pPeak',
# _1tPeak:'1_tPeak',
# _1rPeak:'1_rPeak',
# _1sPeak:'1_sPeak',
# _1qPeak:'1_qPeak',
# _1qrsinterval:'1_qrs_interval',
# _1pqinterval:'1_pq_interval',
# _1qtinterval:'1_qt_interval',
# _1stinterval:'1_st_interval',
# _1qrsmorph0:'1_qrs_morph0',
# _1qrsmorph1:'1_qrs_morph1',
# _1qrsmorph2:'1_qrs_morph2',
# _1qrsmorph3:'1_qrs_morph3',
# _1qrsmorph4:'1_qrs_morph4',}

#     response = requests.get(url, params= info)
#     json = response.json()



# @st.cache_resource
# def load_model():
#     with open('ecg_model/trained_model.pkl', 'rb') as f:
#         model, scaler = pickle.load(f)
#     return model, scaler


def preprocess_input(data, scaler):
    if data.shape[1] != 32:
        st.error("Input data must have 32 columns corresponding to the features used in training.")
        return None
    return scaler.transform(data)

############################### USER INTERFACE #################################

def main():
    st.set_page_config(page_title="ECG Classification App",
                       page_icon="❤️",
                       layout="centered"
    )

    st.markdown("""
    <style>
    .banner {
        width: 100%;
        height: auto;
        margin-bottom: 20px;
    }
    .custom-title {
        font-size: 3em;
        text-align: center;
    }
    .emergency-text {
        text-align: center;
        font-size: 1em;
        margin-top: 50px;
    }
    .emergency-text .highlight {
        color: red;
        font-weight: bold;
    }
    .disclaimer-container {
        background-color: #ffcccc;
        padding: 20px;
        border-radius: 10px;
        color: black;
        text-align: center;
        margin-bottom: 20px;
    }
    .disclaimer-button-container {
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div>
        <img src="https://www.aedsuperstore.com/resources/wp-content/uploads/sites/4/2016/10/Normal-Sinus-Rhythm.gif" class="banner">
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="custom-title">ECG Classification App</h1>', unsafe_allow_html=True)
    st.markdown("""
    Welcome to the ECG Classification App. This tool allows you to upload an ECG file in CSV format and get a classification prediction indicating whether the ECG is **Normal** or **Abnormal (Arrhythmia)**.
    """)

    if "show_disclaimer" not in st.session_state:
        st.session_state.show_disclaimer = True

    if st.session_state.show_disclaimer:
        st.markdown(
        """
        <div class="disclaimer-container">
            <p><strong>Disclaimer</strong>: This app is for educational purposes only and should not be used for medical diagnosis or treatment. Always consult with a healthcare professional for any medical concerns.</p>
        </div>
        """, unsafe_allow_html=True)
        col_center = st.columns([2, 1, 2])[1]
        with col_center:
            if st.button("I Understand", key="acknowledge"):
                st.session_state.show_disclaimer = False
                st.experimental_rerun()
    else:
        uploaded_file = st.file_uploader("Choose a CSV file containing ECG data", type="csv")

        if uploaded_file is not None:
            user_data = pd.read_csv(uploaded_file)

            with st.expander("Show Uploaded Data"):
                st.write(user_data)
            if st.button('Analyze my results'):
                with st.spinner('Analyzing data... ❤️'):
                    # model, scaler = load_model()
                    # processed_data = preprocess_input(user_data, scaler)
                    url = 'https://heartbeat-decoder-7pl35comwq-uk.a.run.app/predict'
                    features = {'_0preRR' : user_data['0_pre-RR'],
                            '_0postRR' : user_data['0_post-RR'],
                            '_0pPeak' : user_data['0_pPeak'],
                            '_0tPeak' : user_data['0_tPeak'],
                            '_0rPeak' : user_data['0_rPeak'],
                            '_0sPeak' : user_data['0_sPeak'],
                            '_0qPeak' : user_data['0_qPeak'],
                            '_0qrsinterval' : user_data['0_qrs_interval'],
                            '_0pqinterval' : user_data['0_pq_interval'],
                            '_0qtinterval' : user_data['0_qt_interval'],
                            '_0stinterval' : user_data['0_st_interval'],
                            '_0qrsmorph0' : user_data['0_qrs_morph0'],
                            '_0qrsmorph1' : user_data['0_qrs_morph1'],
                            '_0qrsmorph2' : user_data['0_qrs_morph2'],
                            '_0qrsmorph3' : user_data['0_qrs_morph3'],
                            '_0qrsmorph4' : user_data['0_qrs_morph4'],
                            '_1preRR' : user_data['1_pre-RR'],
                            '_1postRR' : user_data['1_post-RR'],
                            '_1pPeak' : user_data['1_pPeak'],
                            '_1tPeak' : user_data['1_tPeak'],
                            '_1rPeak' : user_data['1_rPeak'],
                            '_1sPeak' : user_data['1_sPeak'],
                            '_1qPeak' : user_data['1_qPeak'],
                            '_1qrsinterval' : user_data['1_qrs_interval'],
                            '_1pqinterval' : user_data['1_pq_interval'],
                            '_1qtinterval' : user_data['1_qt_interval'],
                            '_1stinterval' : user_data['1_st_interval'],
                            '_1qrsmorph0' : user_data['1_qrs_morph0'],
                            '_1qrsmorph1' : user_data['1_qrs_morph1'],
                            '_1qrsmorph2' : user_data['1_qrs_morph2'],
                            '_1qrsmorph3' : user_data['1_qrs_morph3'],
                            '_1qrsmorph4' : user_data['1_qrs_morph4']
                    }

                    response = requests.get(url, params=features)
                    json = response.json() # previously: 'prediction'


                    if json is not None:

                    # if processed_data is not None:
                    #     prediction = model.predict(processed_data)
                    #     prediction_proba = model.predict_proba(processed_data)

                        prediction_label = 'Normal' if json == 0 else 'Abnormal (Arrhythmia)'
                        # confidence_score = prediction_proba[0][prediction[0]] * 100

                        st.write("## Prediction Result")
                        # st.markdown(f"<h3 style='text-align: center; color: {'green' if prediction[0] == 0 else 'red'};'>{prediction_label} ({confidence_score:.2f}% confidence)</h3>", unsafe_allow_html=True)

                        if json[0] == 0:
                            st.image("https://i.postimg.cc/nhMj7hpL/Normal.png", caption="ECG - Normal Sinus Rhythm", use_column_width=True)
                            st.markdown("""
                            ### What does this mean?
                            **Normal**: The ECG data is classified as normal, indicating that the heart rhythm appears to be regular.

                            **Explanation:**
                            - A normal ECG shows a consistent rhythm and rate, with the heart beating at a regular interval.

                            **Note:**
                            - While a normal ECG is a positive sign, it is essential to remember that this app provides information only and not a professional medical opinion.
                            - If you have any concerns or symptoms, please consult a healthcare professional for a comprehensive evaluation.
                            """)
                        else:
                            st.image("https://i.postimg.cc/br3rDK5b/Arrhythmia.png", caption="ECG - Abnormal (Arrhythmia)", use_column_width=True)
                            st.markdown("""
                            ### What does this mean?
                            **Abnormal (Arrhythmia)**: The ECG data is classified as abnormal, indicating that there may be irregularities in the heart rhythm.
                            - **Arrhythmia** refers to an irregular heart rhythm, which can be too fast, too slow, or erratic.
                            - It is important to consult a healthcare professional for a detailed assessment and diagnosis.

                            **Disclaimer:**
                            - This app provides information only and is not a substitute for professional medical advice, diagnosis, or treatment.
                            - Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
                            """)
                    else:
                        st.error("An error occurred during prediction: Invalid data format")

    st.markdown(
        """
            If you believe you are having a heart attack or a medical emergency, call your local
            <a href="https://en.wikipedia.org/wiki/List_of_emergency_telephone_numbers" class="highlight">emergency services</a>.
        """,
    unsafe_allow_html=True)

if __name__ == "__main__":
    main()
