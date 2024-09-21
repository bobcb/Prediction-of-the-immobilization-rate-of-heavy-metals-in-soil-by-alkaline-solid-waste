#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from joblib import load
import numpy as np

# Load model and scaler
model = load('C:/Users/11931/Desktop/JupyterNotebook/test software/ML_model.joblib')
scaler = load('C:/Users/11931/Desktop/JupyterNotebook/test software/StandardScaler.joblib')

# Set page title
st.title('Prediction of the immobilization rate of heavy metals in soil by alkaline solid waste')

# Apply custom CSS for styling and spacing (without black borders)
st.markdown("""
    <style>
    /* Global font style for the whole app */
    * {
        font-family: 'Arial', sans-serif;
    }

    /* Increase spacing between columns */
    .block-container {
        padding-left: 5rem;
        padding-right: 5rem;
    }

    /* Adjust space between input fields and labels */
    .spacing-row {
        padding-bottom: 2.15em;  
    }

    /* Removed border styling from input fields */
    .stNumberInput input {
        background-color: white !important;
        color: black !important;
    }

    /* Make sure all labels are using the correct font */
    .stTextInput label {
        font-weight: bold !important;
        color: black !important;
    }

    </style>
    """, unsafe_allow_html=True)

# Layout the input fields in three columns with group headers and increased spacing
col1, spacer1, col2, spacer2, col3 = st.columns([1, 0.1, 1, 0.1, 1])

with col1:
    st.markdown("### S/S soil physicochemical properties")
    feature1 = st.number_input(u'$\mathrm{SiO_2\;(\%)}$', step=0.01, format='%.2f')
    feature2 = st.number_input(u'$\mathrm{CaO\;(\%)}$', step=0.01, format='%.2f')
    feature3 = st.number_input(u'$\mathrm{Al_2O_3\;(\%)}$', step=0.01, format='%.2f')
    feature4 = st.number_input(u'$\mathrm{Soil\;pH}$', step=0.01, format='%.2f')
    feature5 = st.number_input(u'$\mathrm{Soil\;heavy\;metal\;concentration\;(mg/kg)}$', step=0.01, format='%.2f')

with col2:
    st.markdown("### Experimental conditions")
    st.markdown('<div class="spacing-row"></div>', unsafe_allow_html=True)
    feature6 = st.number_input(u'$\mathrm{Temperature\;(℃)}$', step=0.01, format='%.2f')
    feature7 = st.number_input(u'$\mathrm{Curing\;time\;(d)}$', step=0.01, format='%.2f')
    feature8 = st.number_input(u'$\mathrm{Liquid/Solid}$', step=0.01, format='%.2f')
    feature9 = st.number_input(u'$\mathrm{Extraction\;agent\;pH}$', step=0.01, format='%.2f')

with col3:
    st.markdown("### Heavy metal properties")
    st.markdown('<div class="spacing-row"></div>', unsafe_allow_html=True)
    feature10 = st.number_input(u'$\mathrm{Electronegativity}$', step=0.01, format='%.2f')
    feature11 = st.number_input(u'$\mathrm{Hydrated\;ion\;radius\;(Å)}$', step=0.01, format='%.2f')

# Experimental heavy metal immobilization rate input
feature = st.number_input(u'$\mathrm{Experimental\;immobilization\;rate\;(\%)}$', step=0.01, format='%.2f')

# Gather all feature inputs
feature_values = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11]

# Prediction and residual calculation
if st.button('Predict'):
    input_data = np.array([feature_values])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)
    residual = abs(float(prediction) - feature)
    
    st.success(f'Predicted Heavy Metal Immobilization Rate: {prediction[0]:.2f}%')
    
    if feature != 0:
        st.success(f'Residual: {residual:.2f}%')


