#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from joblib import load
import numpy as np

# Load model and scaler
model = load('ML_model.joblib')
scaler = load('StandardScaler.joblib')

# Apply custom CSS for styling, alignment, and hyphenation only for the title
st.markdown("""
    <style>
        /* Adjust spacing and align title with columns */
        .block-container {
            padding-left: 7rem;
            padding-right: 7rem;
        }

        /* Set width for the title */
        .main-title {
            font-size: 28px;
            font-weight: bold;
            width: 1020px; /* Set title width */
            word-break: break-word;
            hyphens: auto;  /* Adds hyphenation for long words */
        }

        /* Small header style */
        .small-header {
            font-size: 20px;  /* 设置合适的字号大小 */
            font-weight: bold;  /* 确保加粗 */
        }

        /* Set fixed width for columns */
        .fixed-column {
            width: 400px;  /* Set column width */
        }

        /* Adjust padding in columns */
        .spacing-row {
            padding-bottom: 2.15em;
        }
    </style>
""", unsafe_allow_html=True)

# Set page title with proper alignment and hyphenation for long words
st.markdown('<div class="title-container"><h1 class="main-title">Prediction of the immobilization rate of heavy metals<br>in soil by alkaline solid waste</h1></div>', unsafe_allow_html=True)


# Layout the input fields in three columns with appropriate spacing and alignment
col1, spacer1, col2, spacer2, col3 = st.columns([1, 0.2, 1, 0.2, 1])

with col1:
    st.markdown('<div class="fixed-column">', unsafe_allow_html=True)
    st.markdown('<div class="small-header">S/S soil physicochemical properties</div>', unsafe_allow_html=True)
    feature1 = st.number_input(u'$\mathrm{SiO_2\;(\%)}$', step=0.01, format='%.2f')
    feature2 = st.number_input(u'$\mathrm{CaO\;(\%)}$', step=0.01, format='%.2f')
    feature3 = st.number_input(u'$\mathrm{Al_2O_3\;(\%)}$', step=0.01, format='%.2f')
    feature4 = st.number_input(u'$\mathrm{Soil\;pH}$', step=0.01, format='%.2f')
    feature5 = st.number_input(u'$\mathrm{Soil\;heavy\;metal}$\n$\mathrm{concentration\;(mg/kg)}$', step=0.01, format='%.2f')
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="fixed-column">', unsafe_allow_html=True)
    st.markdown('<div class="small-header">Experimental conditions</div>', unsafe_allow_html=True)
    st.markdown('<div class="spacing-row"></div>', unsafe_allow_html=True)
    feature6 = st.number_input(u'$\mathrm{Temperature\;(℃)}$', step=0.01, format='%.2f')
    feature7 = st.number_input(u'$\mathrm{Curing\;time\;(d)}$', step=0.01, format='%.2f')
    feature8 = st.number_input(u'$\mathrm{Liquid/Solid}$', step=0.01, format='%.2f')
    feature9 = st.number_input(u'$\mathrm{Extraction\;agent\;pH}$', step=0.01, format='%.2f')
    st.markdown('</div>', unsafe_allow_html=True)


with col3:
    st.markdown('<div class="fixed-column">', unsafe_allow_html=True)
    st.markdown('<div class="small-header">Heavy metal properties</div>', unsafe_allow_html=True)
    st.markdown('<div class="spacing-row"></div>', unsafe_allow_html=True)
    feature10 = st.number_input(u'$\mathrm{Electronegativity}$', step=0.01, format='%.2f')
    feature11 = st.number_input(u'$\mathrm{Hydrated\;ion\;radius\;(Å)}$', step=0.01, format='%.2f')
    st.markdown('</div>', unsafe_allow_html=True)

# Gather all feature inputs
feature_values = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11]

# Prediction and residual calculation
if st.button('Predict'):
    input_data = np.array([feature_values])
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)

    st.success(f'Predicted Heavy Metal Immobilization Rate: {prediction[0]:.2f}%')



# In[ ]:





# In[ ]:




