# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 11:24:11 2022

@author: Bruno Tabet
"""

import streamlit as st
import sys
import pandas as pd

# sys.path.append(r"C:\Users\Bruno Tabet\Documents\ENOVA\MVP")

from streamlit_option_menu import option_menu

# from app.dataset import dataset_function
# from app.outliers import outliers_function
# from app.filtering import filtering_function
# from app.regression import regression_function
# from app.results import results_function
# from app.synthetic import synthetic_function
# from app.database import database_function
from PIL import Image
from helpful_funcs.useful_funcs import *

initialization(st)

# st.set_page_config(layout='wide')

# Initialize State


# if 'outliers_removed_manually' not in st.session_state:
#     st.session_state['outliers_removed_manually'] = 0

# if 'outliers_removed_auto' not in st.session_state:
#     st.session_state['outliers_removed_auto'] = 0
    
# for i in range (6):
#     if i not in st.session_state:
#         st.session_state[i] = 0
    


st.title('Smart baseline')
st.text("Let's find the best regression model to predict the baseline !")


if st.session_state['home'] == 0:

    if st.button("Begin !"):
        st.session_state['home'] = 1
        nav_page("Dataset")


if st.session_state['home'] == 1:
    
    if st.button("Continue !"):
        nav_page("Dataset")
    

st.write('')
st.write('')

st.subheader('Explanation of the smart baseline process')

col1, col2 = st.columns(2)

with col1:

    st.write('**Dataset**')
            
    st.write('An excel template is used to collect data that is uploaded to the application. This data includes:')
    st.write('- Baseline timeseries')
    st.write('- Variables that can potentially explain the baseline – these are called features')
    st.write('- Project details')
    
    st.write('The features may have a lower temporal resolution than the baseline as they are automatically aligned to match the baseline’s resolution and bounds.')
    st.write('Weather-related features are automatically included without user input.')
    st.write('The baseline is automatically normalized hourly and the user can remove abnormal points from the dataset.')
    
    st.write('**Synthetic features**')
    st.write('The user has the capability to increase the number of features by computing their inverse, square and/or square root.')
    
    st.write('**Feature removal**')
    st.write('Features that are not correlated to the baseline can be automatically removed using “filters”. Manual feature removal is also available for the user.')
    
    
    st.write('**Regressions**')
    st.write('The user can choose the maximum number of features that should be included in the regressions. All the combinations of features that are not correlated are computed and regression and cross validation methods are run on these combinations.')
    
    
    st.write('**Results**')
    st.write('The results of all the regressions can be viewed and easily compared. The user can fine-tune selected models by removing outliers.')
    
    
    st.write('**Database**')
    st.write('Once the user has selected the final model, the project details and model outputs are added to a database.')
    
    

with col2:
    
    image = Image.open('Process of the app.png')
    st.image(image, caption  = 'Process of the app')