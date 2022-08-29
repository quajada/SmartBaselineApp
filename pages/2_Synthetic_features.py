# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 11:57:52 2022

@author: Bruno Tabet
"""


import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.api as sm
import os
import sys
import matplotlib.pyplot as plt
import plotly_express as px


# sys.path.append(r"C:\Users\Bruno Tabet\Documents\ENOVA\MVP")

from features.synthetic_features import SyntheticFeatures, CDD, HDD
from cleandata.cleandata import Aggregator, CleanColumns, CleanRows
from engines.engine import Engine
from helpful_funcs.excel_funcs import ReadExcel
from combinations.combinations import Combinations
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
from streamlit_plotly_events import plotly_events
from helpful_funcs.useful_funcs import *


initialization(st)


st.header('Create synthetic features')

sqrt_and_squared = False
products = False

if st.session_state['synthetic_features_created'] == 0:
    
    st.write('Before creating the synthetic features, you have to upload your data.')


if st.session_state['synthetic_features_created'] == 1:
    
    
    st.session_state['x_df_synthetic'] = st.session_state['x_df_outliers'].copy()
    st.session_state['y_df_synthetic'] = st.session_state['y_df_outliers'].copy()
    
    synth = SyntheticFeatures(st.session_state['x_df_synthetic'])
    
    if 'YES' == st.selectbox('Do you want to create the squared root and the square of the features ?',
                              options = ['NO','YES']):
        sqrt_and_squared = True
        
        
    # if 'YES' == st.selectbox('Do you want to create the synthetic features corresponding to the products of the original features (this could take a few minutes) ?',
    #                           options = ['NO', 'YES']):
    #     products = True
    
    inv_columns = []
    for column in st.session_state['x_df_synthetic'].columns:
        if st.session_state['x_df_synthetic'][st.session_state['x_df_synthetic'][column] == 0].empty:
            inv_columns.append(column)
        
    st.session_state['inverse_features'] = st.multiselect('Which features do you want to compute the inverse of ?',
                                      options = [feature for feature in inv_columns])
    
    st.session_state['strings_to_remove'] = st.multiselect('Do you want to remove the max, the min or the std of features ? Select those you want to remove',
                                                           options = ['max', 'min', 'std'])    
    
    st.write('')
    st.write('')
    st.write('')
    

    if st.button('COMPUTE'):
        
        with st.spinner('Creating the synthetic features and removing features containing 3 equal values.'):
            if sqrt_and_squared:
                synth.create_sqrt_and_squared()
                
            # if products:
            #     synth.create_products()
                    
            if len(st.session_state['inverse_features']) > 0:
                synth.create_inverse(st.session_state['inverse_features'])
            
            # clean = CleanColumns(st.session_state['x_df_synthetic'])
            # st.session_state['x_df_synthetic'] = clean.remove_bad_columns()
            
            for string in st.session_state['strings_to_remove']:
                string = string + '_'
                synth.remove_str(string)
            
            st.session_state['x_df_synthetic'] = synth.x_df
            
            st.session_state['filters_applied'] = 1
            st.session_state['synthetic_features_created'] = 2
            st.experimental_rerun()
            
    
if st.session_state['synthetic_features_created'] == 2:
    
    st.write('The synthetic features have been successfully created.')
    
    if st.checkbox('Show all the features and the normalized baseline', value = True, key = 5):
        col_x, col_y = st.columns(2)
        with col_x:
            st.dataframe(st.session_state['x_df_synthetic'])
            st.write(str(len(st.session_state['x_df_synthetic'])) + " rows, " + str(len(st.session_state['x_df_synthetic'].columns))+ " columns")
        with col_y:
            st.dataframe(st.session_state['y_df_synthetic'])

    st.write('')

 
    st.write('Do you want to change the synthetic features ? Everything you have done afterwards will be lost.')
    if st.button('CHANGE'):
        st.session_state['synthetic_features_created'] = 1
        st.session_state['filters_applied'] = 0
        st.session_state['filters_manual'] = 0
        st.session_state['regression_done'] = 0
        st.session_state['results'] = 0
        st.session_state['database'] = 0
        initialize_results_df_outliers(st)
        initialize_selections(st)
        delete_selected(st)
        st.experimental_rerun()
    
        
    st.write('')
    st.write('')

    col1, col2, col3 = st.columns([1, 5, 1])     
    
    with col1:
        if st.button("< Prev"):
            nav_page('Dataset')

    with col3:
        if st.button("Next >"):
            nav_page("Feature_removal")
