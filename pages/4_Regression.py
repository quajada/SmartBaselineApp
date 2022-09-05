# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:04:10 2022

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
from streamlit_plotly_events import plotly_events

# sys.path.append(r"C:\Users\Bruno Tabet\Documents\ENOVA\MVP")

from math import *
from features.synthetic_features import SyntheticFeatures, CDD, HDD
from cleandata.cleandata import Aggregator, CleanColumns, CleanRows
from engines.engine import Engine
from helpful_funcs.excel_funcs import ReadExcel
from combinations.combinations import Combinations
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
from helpful_funcs.useful_funcs import *


initialization(st)

if 'name_sidebar' in st.session_state:
    st.sidebar.title("Project name : " + st.session_state['name_sidebar'])


st.header('Regression')

if st.session_state['regression_done'] == 0:
    
    st.write('Before computing the regression, you have to finish applying the filters to remove useless features.')


if st.session_state['regression_done'] == 1:
    
    st.session_state['x_df_regression'] = st.session_state['x_df_filters'].copy()
    st.session_state['y_df_regression'] = st.session_state['y_df_filters'].copy()
    

    st.write("The algorithm will create all the possible combinations of the features and exclude those where the features are highly correlated between themselves, to minimize over-fitting.")
    st.write("Cross validation is used to evaluate how well the model generalizes and predicts unseen data, which gives insight into the risk and uncertainty of each model. Cross validation is particularly relevant with large datasets.")
    
    st.write('')
    # st.write('Choose the parameters you want for the regression.')
    
    # col_1, col_2 = st.columns(2)
    
    st.session_state['max_features_str'] = st.selectbox('Number of features for each model:', 
                                    options = ["1", "up to 2", "up to 3", "up to 4"])

    nbs = ['1', '2', '3', '4']
    
    for i in range (len(nbs)):
        if nbs[i] in st.session_state['max_features_str']:
            st.session_state['max_features'] = i+1
    
    # st.session_state['criteria'] = st.selectbox('What criteria do you want to choose for selecting the best models?', 
    #                                 options = ["std_dev", "r2"])
 
    # st.session_state['nb_top'] = st.number_input('How many best results do you want to see?',
    #                                   min_value = 1, max_value = 100, value = 10, step = 1)
    
    st.session_state['nb_folds'] = st.number_input('How many folds (or iterations) do you want? If you increase the number of folds, the calculation will take longer, but you will evaluate more scenarios.',
                                                 min_value = 2, max_value = 200, value = 100, step =1)
    
    
    st.session_state['test_size_chosen'] = st.slider('For each fold, the data will be split into training and testing sets. What percentage of your dataset do you want to use for testing?'
                                                       +" Note that the minimum number of test points is your number of features + 2.",
                                                       min_value = 1, max_value = 50,  value = 33, step = 1)
  
    st.session_state['test_size'] = max(st.session_state['max_features']+2, (ceil(st.session_state['test_size_chosen']*len(st.session_state['x_df_regression']) / 100)))
    
    st.markdown("You have chosen **" + str(st.session_state['test_size']) + " out of "+ str(len(st.session_state['x_df_regression']))+ "** data points for the test points.")
    
    st.write('')
    st.write('')
    
    # st.write('Are you ready to compute the combinations and the regressions ? This might take a few minutes ...')

    if st.session_state['test_size'] <= st.session_state['max_features'] + 1:
         st.button('Compute !', disabled = True)
         st.write("You can't compute the regression since you have more features in some of your combinations than data points in your test size.")
    
    else:        
        if len(st.session_state['x_df_regression'].columns) == 0:
            st.write('**You need to have at least one feature in order to compute the regression. Perhaps you should change the paramaters chosen for feature removal.**')
            st.button('COMPUTE', disabled = True)
        
        else:
            col_1, col_2, col_3 = st.columns([0.7, 2, 5])
            col_2.write('This might take a few minutes ...')
            if col_1.button('COMPUTE'):
            
                with st.spinner('Computing the combinations and the regressions'):
                    combi = Combinations(st.session_state['x_df_regression'].columns, st.session_state['max_features'])
                    combinations = combi.compute_combinations(st.session_state['x_df_regression'])
                    
                    final = Engine(st.session_state['x_df_regression'], st.session_state['y_df_regression'], combinations, max_variables = st.session_state['max_features'], nb_folds = st.session_state['nb_folds'], test_size = st.session_state['test_size'])
                    final.compute_cross_validation()
                    # final.are_combinations_IPMVP_consistently_compliant()
                    st.session_state['results_dict'] = final.are_combinations_IPMVP_compliant()
                    st.session_state['results_df'] = final.get_df_results()
                    
                    
                    for combination in combinations:
                        st.session_state['x_df_regression'+str(combination)] = st.session_state['x_df_regression'].copy()
                        st.session_state['y_df_regression'+str(combination)] = st.session_state['y_df_regression'].copy()
                                    
                    st.session_state['results'] = 1
                    st.session_state['database'] = 1
                    st.session_state['regression_done'] = 2
                    nav_page("Results")
                    # st.experimental_rerun()


if st.session_state['regression_done'] == 2:
    
    # col_1, col_2 = st.columns(2)
    
    st.selectbox('Number of features for each model:', options = [st.session_state['max_features_str']], disabled = True)
    
    st.number_input('How many folds (or iterations) do you want? If you increase the number of folds, the calculation will take longer, but you will evaluate more scenarios.', min_value = 2, max_value = 200, value = st.session_state['nb_folds'], step =1, disabled = True)
    
    st.slider('For each fold, the data will be split into training and testing sets. What percentage of your dataset do you want to use for testing?'+
              " Note that the minimum number of test points is your number of features + 2.",
              min_value = 1, max_value = 50,  value = st.session_state['test_size_chosen'], step = 1, disabled = True)


    st.write('')
    st.write('')
    
    st.write('Do you want to change the parameters of your regressions ? Everything you have done afterwards will be lost.')
    if st.button('CHANGE'):
        st.session_state['regression_done'] = 1
        st.session_state['results'] = 0
        initialize_results_df_outliers(st)
        st.session_state['database'] = 0
        initialize_selections(st)
        delete_selected(st)
        st.experimental_rerun()
            
    st.write('')
    st.write('')   
    st.write('')
    st.write('')    

    
    
    col1, col2, col3 = st.columns([1, 5, 1])        

    with col1:
        if st.button("< Prev"):
            nav_page('Feature_removal')

    with col3:
        if st.button("Next >"):
            nav_page("Results")