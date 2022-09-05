# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:00:41 2022

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

from features.synthetic_features import SyntheticFeatures, CDD, HDD
from cleandata.cleandata import Aggregator, CleanColumns, CleanRows
from filters.filterdata import FilterData
from engines.engine import Engine
from helpful_funcs.excel_funcs import ReadExcel
from combinations.combinations import Combinations
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
from helpful_funcs.useful_funcs import *
from PIL import Image


initialization(st)

if 'name_sidebar' in st.session_state:
    st.sidebar.title("Project name : " + st.session_state['name_sidebar'])

st.header('Feature removal')

st.subheader('Automatic')    

if st.session_state['filters_applied'] == 0:
    
    st.write('Before applying the filters, you have to create the synthetic features you want to add.')
    

if st.session_state['filters_applied'] == 1:
    

    st.write('The user can automatically remove features using statistical filters : a threshold of 0 removes no features, and a threshold of 1 will remove all of them')

    if st.checkbox('Show more information about the thresholds:'):
        image = Image.open('Correlation.png')
        st.image(image, caption  = 'Scale of the correlation coefficients')
    

    st.session_state['x_df_filters'] = st.session_state['x_df_synthetic'].copy()
    st.session_state['y_df_filters'] = st.session_state['y_df_synthetic'].copy()
    
    st.write('**Pearson**')

    st.session_state['pearson'] = st.slider("What threshold do you want to chose for the Pearson filter :  put 0 if you don't want to use this filter, and 1 if you want to remove all features. The coefficient identifies linear relations between the two features.", 
                        min_value = 0.00, max_value = 1.00, value = 0.39, step = 0.01, key = 19890389089408902890842)
    
    st.write('**Spearman**')
    
    st.session_state['spearman'] = st.slider("What threshold do you want to chose for the Spearman filter :  put 0 if you don't want to use this filter, and 1 if you want to remove all features. The coefficient identifies linear and monotonic relations between the two features.", 
                        min_value = 0.00, max_value = 1.00, value = 0.39, step = 0.01, key = 12987837189278912)
     
    st.write('**Mutual information**')
    
    st.session_state['info'] = st.slider("What threshold do you want to chose for the mutual information filter :  put 0 if you don't want to use this filter, and 1 if you want to remove all features. The coefficient identifies the information that can be understood from a feature with the other feature.", 
                        min_value = 0.00, max_value = 1.00, value = 0.39, step = 0.01, key = 1989018790789127)
    
    st.write('')

    if st.button('REMOVE FEATURES', key =5):
        
        with st.spinner('Appying the filters'):
            
            filt = FilterData(st.session_state['x_df_filters'], st.session_state['y_df_filters'])
            
            st.session_state['bad_features_pearson'] = filt.get_bad_features_pearson(st.session_state['pearson'])
            filt.remove_features(st.session_state['bad_features_pearson'])

            st.session_state['bad_features_spearman'] = filt.get_bad_features_spearman(st.session_state['spearman'])
            filt.remove_features(st.session_state['bad_features_spearman'])
            
            st.session_state['bad_features_info'] = filt.get_bad_features_info(st.session_state['info'])
            filt.remove_features(st.session_state['bad_features_info'])
            
            # bad_other_features = filt.get_worst_features()
            # filt.remove_features(bad_other_features)
            st.session_state['x_df_filters'] = filt.x_df
            
            st.session_state['filters_manual'] = 1
            st.session_state['filters_applied'] = 2
            st.session_state['remaining_features'] = [feature for feature in st.session_state['x_df_filters'].columns]
            delete_selected(st)
            st.experimental_rerun()

if st.session_state['filters_applied'] == 2:
    
    
    st.write('The filters have been applied successfully !')
    
    if st.checkbox('Show features removed by Pearson.'):
        st.markdown(st.session_state['bad_features_pearson'])
    
    if st.checkbox('Show features removed by Spearman.'):
        st.markdown(st.session_state['bad_features_spearman'])
        
    if st.checkbox('Show features removed by the mutual information.'):
        st.markdown(st.session_state['bad_features_info'])

    if st.checkbox('Show remaining features.', value = True):
        st.markdown('You have '+ str(len(st.session_state['remaining_features']))+ ' remaining features :')
        st.markdown(st.session_state['remaining_features'])
        
        
        
    st.write('')
    st.write('')
 
    
    st.write('Do you want to change the thresholds ? Everything you have done afterwards will be lost.')
    
    if st.button('CHANGE'):
        st.session_state['filters_applied'] = 1
        st.session_state['filters_manual'] = 0
        st.session_state['regression_done'] = 0
        st.session_state['results'] = 0
        st.session_state['results_df_outliers'] = pd.DataFrame(columns = ['combinations', 'r2', 'std_dev', 'r2_cv_test', 'std_dev_cv', 'intercept', 'pval', 'tval', 'IPMVP_compliant', 'AIC', 'AIC_adj', 'size', 'nb_data_points', 'nb_outliers_removed', 'version'])
        st.session_state['database'] = 0
        initialize_results_df_outliers(st)
        initialize_selections(st)
        delete_selected(st)
        st.experimental_rerun()
            
    st.write('')
    st.write('')   
    st.write('')
    st.write('')
            


st.subheader('Manual')

if st.session_state['filters_manual'] == 0:
    
    st.write('Before removing the features manually, you have to apply the correlation filters implemented.')


if st.session_state['filters_manual'] == 1:
            
    st.write('You now have the possibility to remove features manually.')
    
    if st.checkbox('Do you want to plot some of the features before deciding which ones to remove ?'):
        
        list_of_options = [feature for feature in st.session_state['x_df_filters'].columns]
        list_of_options.sort()  
        
        
        feature_chosen = st.selectbox('Choose a feature you want to plot',
                                      options = list_of_options)
        
        col_3, col_4 = st.columns(2)
        
        with col_3:
            fig = plt.figure()
            plt.scatter(st.session_state['x_df_filters'][feature_chosen], st.session_state['y_df_filters'].values, color = 'blue')
            plt.ylabel('Baseline')
            plt.xlabel(str(feature_chosen))
            plt.title('Baseline as a function of '+ str(feature_chosen))
            st.pyplot(fig)
            
        
    st.session_state['bad_features_chosen'] = st.multiselect('Choose the features you want to remove manually',
                                      options = [feature for feature in st.session_state['x_df_filters'].columns])
            
    if st.button('REMOVE FEATURES', key = 6):
        
        with st.spinner('Removing the features chosen'):
            
            st.session_state['x_df_filters'] = st.session_state['x_df_filters'].drop([feature for feature in st.session_state['bad_features_chosen']], axis = 1)

            st.session_state['regression_done'] = 1
            st.session_state['filters_manual'] = 2
            st.experimental_rerun()   

if st.session_state['filters_manual'] == 2:
    
    st.write('You have successfully removed the selected features !')

    if st.checkbox('Show features removed manually.'):
        st.markdown(st.session_state['bad_features_chosen'])

    if st.checkbox('Show remaining features.', value = True, key = 10000000001767):
        st.markdown('You have '+ str(len(st.session_state['x_df_filters'].columns))+ ' remaining features :')
        st.markdown([feature for feature in st.session_state['x_df_filters'].columns])

    st.write('')
    st.write('')

    if st.checkbox('Do you stil want to plot features ?'):
        
        list_of_options = [feature for feature in st.session_state['x_df_filters'].columns]
        list_of_options.sort()  
        
        feature_chosen = st.selectbox('Choose a feature you want to plot',
                                      options = list_of_options)
        
        col_3, col_4 = st.columns(2)
        
        with col_3:
            fig = plt.figure()
            plt.scatter(st.session_state['x_df_filters'][feature_chosen], st.session_state['y_df_filters'].values, color = 'blue')
            plt.ylabel('Baseline')
            plt.xlabel(str(feature_chosen))
            plt.title('Baseline as a function of '+ str(feature_chosen))
            st.pyplot(fig)
    
    
    st.write('')
    st.write('')
 
    
    st.write('Do you want to change the features you have removed ? Everything you have done afterwards will be lost.')
    
    if st.button('CHANGE', 'filters_manually_removed'):
        st.session_state['filters_manual'] = 1
        st.session_state['regression_done'] = 0
        st.session_state['results'] = 0
        st.session_state['database'] = 0
        initialize_results_df_outliers(st)
        initialize_selections(st)
        st.experimental_rerun()
            
    st.write('')
    st.write('')   
    st.write('')
    st.write('')
            
    col1, col2, col3 = st.columns([1, 5, 1])        

    with col1:
        if st.button("< Prev"):
            nav_page('Synthetic_features')

    with col3:
        if st.button("Next >"):
            nav_page("Regression")
            
    