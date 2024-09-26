#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from joblib import load
import numpy as np

# Load model and scaler
model = load('ML_model.joblib')
scaler = load('StandardScaler.joblib')



# Set page title with proper alignment and hyphenation for long words
st.title('Prediction of the immobilization rate of heavy metals in soil by alkaline solid waste)

# Create a container for all the content

col1,  col2, col3 = st.columns(3)

    with col1:
        
        
        feature1 = st.number_input(u'$\mathrm{SiO_2\;(\%)}$', step=0.01, format='%.2f')
        feature2 = st.number_input(u'$\mathrm{CaO\;(\%)}$', step=0.01, format='%.2f')
        feature3 = st.number_input(u'$\mathrm{Al_2O_3\;(\%)}$', step=0.01, format='%.2f')
        feature4 = st.number_input(u'$\mathrm{Soil\;pH}$', step=0.01, format='%.2f')
        feature5 = st.number_input(u'$\mathrm{Soil\;heavy\;metal}$\n$\mathrm{concentration\;(mg/kg)}$', step=0.01, format='%.2f')
        

    with col2:
        
        feature6 = st.number_input(u'$\mathrm{Temperature\;(℃)}$', step=0.01, format='%.2f')
        feature7 = st.number_input(u'$\mathrm{Curing\;time\;(d)}$', step=0.01, format='%.2f')
        feature8 = st.number_input(u'$\mathrm{Liquid/Solid}$', step=0.01, format='%.2f')
        feature9 = st.number_input(u'$\mathrm{Extraction\;agent\;pH}$', step=0.01, format='%.2f')
        

    with col3:
        
        feature10 = st.number_input(u'$\mathrm{Electronegativity}$', step=0.01, format='%.2f')
        feature11 = st.number_input(u'$\mathrm{Hydrated\;ion\;radius\;(Å)}$', step=0.01, format='%.2f')
        

feature = st.number_input(u'$\mathrm{Experimental\;heavy\;metal\;immobilization\;rate\;(\%)}$', step=0.01, format='%.2f')
        

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

