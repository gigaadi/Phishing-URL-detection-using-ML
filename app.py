import streamlit as st
import numpy as np
import pandas as pd
from sklearn import metrics 
import pickle
from feature import FeatureExtraction

# Load the model
file = open("model.pkl", "rb")
gbc = pickle.load(file)
file.close()

# Streamlit App
st.title("Phishing Attack Detection")

# Input URL from user
url = st.text_input("Enter the URL:")
if url:
    obj = FeatureExtraction(url)
    x = np.array(obj.getFeaturesList()).reshape(1, 30)

    # Prediction
    y_pred = gbc.predict(x)[0]
    y_pro_phishing = gbc.predict_proba(x)[0, 0]
    y_pro_non_phishing = gbc.predict_proba(x)[0, 1]

    # Display result
    if y_pred == 1:
        st.success(f"The URL is safe to go.")
    else:
        st.warning(f"The URL is unsafe.")
