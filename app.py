#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from joblib import load
import numpy as np

# Load model and scaler
model = load('ML_model.joblib')
scaler = load('StandardScaler.joblib')

# Set page title
st.title('Prediction of the immobilization rate of heavy metals in soil by alkaline solid waste')

# Apply custom CSS for styling and spacing (without black borders)
st.markdown("""
    <style>
    /* Increase spacing between columns */
    .block-container {
        padding-left: 7rem;  /* Adjust padding to make the whole layout wider */
        padding-right: 7rem;
    }

    /* Add spacing row */
    .spacing-row {
        padding-bottom: 2.15em;  /* Adjust this value to control the spacing */
    }

    /* Custom font size for the column headers */
    .small-header {
        font-size: 22px;  /* You can adjust this size as needed */
        font-weight: bold;
        padding-bottom: 10px; /* Adds some space below the title */
    }
    </style>
    """, unsafe_allow_html=True)

# Layout the input fields in three columns with increased spacing between columns
col1, spacer1, col2, spacer2, col3 = st.columns([1.2,0.5, 1.2, 0.5, 1.2])

with col1:
    st.markdown('<div class="small-header">S/S soil physicochemical properties</div>', unsafe_allow_html=True)
    feature1 = st.number_input(u'$\mathrm{SiO_2\;(\%)}$', step=0.01, format='%.2f')
    feature2 = st.number_input(u'$\mathrm{CaO\;(\%)}$', step=0.01, format='%.2f')
    feature3 = st.number_input(u'$\mathrm{Al_2O_3\;(\%)}$', step=0.01, format='%.2f')
    feature4 = st.number_input(u'$\mathrm{Soil\;pH}$', step=0.01, format='%.2f')
    feature5 = st.number_input(u'$\mathrm{Soil\;heavy\;metal}$\n$\mathrm{concentration\;(mg/kg)}$', step=0.01, format='%.2f')

with col2:
    st.markdown('<div class="small-header">Experimental conditions</div>', unsafe_allow_html=True)
    # Add spacing before inputs in this column
    st.markdown('<div class="spacing-row"></div>', unsafe_allow_html=True)
    feature6 = st.number_input(u'$\mathrm{Temperature\;(℃)}$', step=0.01, format='%.2f')
    feature7 = st.number_input(u'$\mathrm{Curing\;time\;(d)}$', step=0.01, format='%.2f')
    feature8 = st.number_input(u'$\mathrm{Liquid/Solid}$', step=0.01, format='%.2f')
    feature9 = st.number_input(u'$\mathrm{Extraction\;agent\;pH}$', step=0.01, format='%.2f')

with col3:
    st.markdown('<div class="small-header">Heavy metal properties</div>', unsafe_allow_html=True)
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

