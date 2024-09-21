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
        font-size: 22px;  /* Adjust the size for section headers */
        font-weight: bold;
        padding-bottom: 10px; /* Adds some space below the title */
    }

    /* Custom font size for the feature labels */
    .feature-label {
        font-size: 20px;  /* Adjust the size for feature labels */
    }
    </style>
    """, unsafe_allow_html=True)

# Layout the input fields in three columns with increased spacing between columns
col1, spacer1, col2, spacer2, col3 = st.columns([1.2, 0.5, 1.2, 0.5, 1.2])

with col1:
    st.markdown('<div class="small-header">S/S soil physicochemical properties</div>', unsafe_allow_html=True)
    feature1 = st.number_input('<span class="feature-label">SiO<sub>2</sub> (%)</span>', step=0.01, format='%.2f', help=u'$\mathrm{SiO_2\;(\%)}$')
    feature2 = st.number_input('<span class="feature-label">CaO (%)</span>', step=0.01, format='%.2f', help=u'$\mathrm{CaO\;(\%)}$')
    feature3 = st.number_input('<span class="feature-label">Al<sub>2</sub>O<sub>3</sub> (%)</span>', step=0.01, format='%.2f', help=u'$\mathrm{Al_2O_3\;(\%)}$')
    feature4 = st.number_input('<span class="feature-label">Soil pH</span>', step=0.01, format='%.2f', help=u'$\mathrm{Soil\;pH}$')
    feature5 = st.number_input('<span class="feature-label">Soil heavy metal concentration (mg/kg)</span>', step=0.01, format='%.2f', help=u'$\mathrm{Soil\;heavy\;metal\;concentration\;(mg/kg)}$')

with col2:
    st.markdown('<div class="small-header">Experimental conditions</div>', unsafe_allow_html=True)
    st.markdown('<div class="spacing-row"></div>', unsafe_allow_html=True)
    feature6 = st.number_input('<span class="feature-label">Temperature (℃)</span>', step=0.01, format='%.2f', help=u'$\mathrm{Temperature\;(℃)}$')
    feature7 = st.number_input('<span class="feature-label">Curing time (d)</span>', step=0.01, format='%.2f', help=u'$\mathrm{Curing\;time\;(d)}$')
    feature8 = st.number_input('<span class="feature-label">Liquid/Solid</span>', step=0.01, format='%.2f', help=u'$\mathrm{Liquid/Solid}$')
    feature9 = st.number_input('<span class="feature-label">Extraction agent pH</span>', step=0.01, format='%.2f', help=u'$\mathrm{Extraction\;agent\;pH}$')

with col3:
    st.markdown('<div class="small-header">Heavy metal properties</div>', unsafe_allow_html=True)
    feature10 = st.number_input('<span class="feature-label">Electronegativity</span>', step=0.01, format='%.2f', help=u'$\mathrm{Electronegativity}$')
    feature11 = st.number_input('<span class="feature-label">Hydrated ion radius (Å)</span>', step=0.01, format='%.2f', help=u'$\mathrm{Hydrated\;ion\;radius\;(Å)}$')

# Experimental heavy metal immobilization rate input
feature = st.number_input('<span class="feature-label">Experimental immobilization rate (%)</span>', step=0.01, format='%.2f', help=u'$\mathrm{Experimental\;immobilization\;rate\;(\%)}$')

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

