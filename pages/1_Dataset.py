# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 11:40:25 2022

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
import time
import datetime
from datetime import datetime


# sys.path.append(r"C:\Users\Bruno Tabet\Documents\ENOVA\MVP")

from features.synthetic_features import SyntheticFeatures, CDD, HDD
from cleandata.cleandata import Aggregator, CleanColumns, CleanRows
from helpful_funcs.excel_funcs import ReadExcel
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
from helpful_funcs.useful_funcs import *

initialization(st)

if 'name_sidebar' in st.session_state:
    st.sidebar.title("Project name : " + st.session_state['name_sidebar'])


def rename(x_df):
    
    names = {}
    
    features = [feature for feature in x_df.columns]
    
    for feature in features:
        
        if feature == 'temp':
            names[feature] = 'temperature'
        if feature == 'dwpt':
            names[feature] = 'dew_point_temperature'
        if feature == 'rhum':
            names[feature] = 'relative_humidity'
        if feature == 'wdir':
            names[feature] = 'wind_direction'
        if feature == 'wspd':
            names[feature] = 'wind_speed'
        if feature == 'pres':
            names[feature] = 'pressure'
        if feature == 'coco':
            names[feature] = 'coco'
        if feature == 'humratio':
            names[feature] = 'humidity_ratio'
        if feature == 'wbtemp':
            names[feature] = 'wet-bulb_temperature'
        if feature == 'dptemp':
            names[feature] = 'dptemp'
        if feature == 'ppwvap':
            names[feature] = 'partial_pressure_of_water_vapor'
        if feature == 'enthma':
            names[feature] = 'enthalpy'
        if feature == 'spvolma':
            names[feature] = 'specific_volume_of_moist_air'
        if feature == 'degsat':
            names[feature] = 'degree_of_saturation'
        if feature not in names:
            names[feature] = feature
        
    for column in x_df.columns:
        x_df.rename(columns = {column : names[column]}, inplace = True)
        
    return x_df




st.header('The dataset')

st.subheader('Upload file')

if st.session_state['file_uploaded'] == 0:

    # path = r"C:\Users\Bruno Tabet\Documents\ENOVA\Input_template.xlsx"
    
    file = st.file_uploader('Upload your excel file', type = ['xlsx'], key = 1)
    
    if file is not None:
        
        st.session_state['file'] = file
        st.session_state['data_uploaded'] = 1
        st.session_state['file_uploaded'] = 2
        st.experimental_rerun()
        
        
if st.session_state['file_uploaded'] == 2:
    st.file_uploader('Upload your excel file', type = ['xlsx'], disabled = True, key=2)
    st.write('File uploaded successfully !')
    st.write("We removed the NaN values from the dataframe, and we made sure that there weren't any duplicates in the columns.")

    if st.session_state['data_uploaded'] == 1:

        st.session_state['excel'] = ReadExcel(st.session_state['file'])
        
        st.session_state['name_sidebar'] = st.session_state['excel'].data['Project name']
        x_df, y_df, baseline = st.session_state['excel'].preprocess_data(st.session_state['file'])
        
        clean = CleanColumns(x_df)
        clean.remove_nan()
        clean.fill_nan()
        clean.remove_duplicates()
        
        x_df = clean.x_df
        
        st.session_state['sheet'] = st.session_state['excel'].wb['Project']
        st.session_state['x_df_dataset'] = x_df
        st.session_state['y_df_with_dates'] = y_df[['From (incl)', 'To (excl)', 'Normalized baseline']]
        st.session_state['y_df_with_dates']['Normalized baseline'] = y_df['Normalized baseline'].astype(np.float64)
        st.session_state['y_df_with_dates']['Baseline'] = baseline[baseline.columns[-1]].astype(np.float64)
        st.session_state['y_df_dataset'] = st.session_state['y_df_with_dates'].copy()
        st.session_state['y_df_dataset']['Timedelta'] = y_df['Timedelta']
        
        rename(st.session_state['x_df_dataset'])
        
        st.session_state['synthetic_features_created'] = 1
        st.session_state['data_uploaded'] = 2
        st.session_state['outliers_removal'] = 1
        initialize_selected_outliers_points(st)
        st.experimental_rerun()

    if st.session_state['data_uploaded'] == 2:
                
        if st.checkbox('Show the features and baseline', value = True):
            col_x, col_y = st.columns(2)
            with col_x:
                st.dataframe(st.session_state['x_df_dataset'])
                st.write(str(len(st.session_state['x_df_dataset'])) + " rows, " + str(len(st.session_state['x_df_dataset'].columns))+ " columns")
            with col_y:
                st.dataframe(st.session_state['y_df_dataset'])
            
        
        st.write('Do you want to pick another file ? Everything you have done afterwards will be lost.')
        
        if st.button('CHANGE', key = 'upload'):

            st.session_state['file_uploaded'] = 0
            st.session_state['outliers_removal'] = 0
            st.session_state['selected_points'] = {}
            st.session_state['selected_pt'] = []
            st.session_state['synthetic_features_created'] = 0
            st.session_state['filters_applied'] = 0
            st.session_state['filters_manual'] = 0
            st.session_state['regression_done'] = 0
            st.session_state['results'] = 0
            initialize_results_df_outliers(st)
            initialize_selections(st)
            initialize_selected_outliers_points(st)
            delete_selected(st)
            st.experimental_rerun()    
        
        
st.subheader('Dataset selection')

if st.session_state['outliers_removal'] == 0:
    st.write('You have to upload your excel file first.')


if st.session_state['outliers_removal'] == 1:
    
    st.session_state['x_df_outliers'] = st.session_state['x_df_dataset'].copy()
    st.session_state['y_df_outliers'] = st.session_state['y_df_dataset'].copy()
            
    st.write("Click on the points you want to remove. If you misclicked on a point, just click on it again.")

    selected_date = [st.session_state['selected_points'][x]['date'] for x in st.session_state['selected_points']]
    selected_y = [st.session_state['selected_points'][x]['y'] for x in st.session_state['selected_points']]
    
    outliers_date = [st.session_state['outliers_points'][x]['date'] for x in st.session_state['outliers_points']]
    outliers_y = [st.session_state['outliers_points'][x]['y'] for x in st.session_state['outliers_points']]


    plot = px.scatter()
    plot.add_scatter(x= selected_date, y = selected_y, mode = 'markers', marker = dict(color = 'green'), name = 'points to keep')
    plot.add_scatter(x = outliers_date, y = outliers_y, mode = 'markers', marker = dict(color = 'red'), name = 'points to remove')
    plot.update_layout(title = {'text' : 'Baseline as a function of time','x':0.47, 'xanchor': 'center', 'yanchor': 'top'},
                       xaxis_title ='Time', yaxis_title='Baseline')
    st.session_state['selected_pt'] = plotly_events(plot, click_event = True, key = st.session_state['iter'])
    
    st.write('')
        
        
    if st.session_state['selected_pt'] != []:
        date_str = st.session_state['selected_pt'][0]['x']
        date = datetime_object = datetime.strptime(date_str, '%Y-%m-%d')
        y = float(st.session_state['selected_pt'][0]['y'])

        for key in st.session_state['selected_points']:
            if st.session_state['selected_points'][key]['y'] == y and st.session_state['selected_points'][key]['date'] == date:
                x = key

        for key in st.session_state['outliers_points']:
            if st.session_state['outliers_points'][key]['y'] == y and st.session_state['outliers_points'][key]['date'] == date:
                x = key


        if x in st.session_state['selected_points']:
            del st.session_state['selected_points'][x]
            st.session_state['outliers_points'][x] = {}
            st.session_state['outliers_points'][x]['y'] = y
            st.session_state['outliers_points'][x]['date'] = date
            
        else:
            st.session_state['selected_points'][x] = {}
            st.session_state['selected_points'][x]['y'] = y
            st.session_state['selected_points'][x]['date'] = date
            del st.session_state['outliers_points'][x]
        
        st.session_state['iter'] += 1
        st.session_state['selected_pt'] = []
        st.experimental_rerun()
    
    
    if len(st.session_state['selected_points']) < 7:
        st.write("**The smart baseline won't work with less than 7 data points. You are currently keeping only "+ str(len(st.session_state['selected_points']))+" data points.**")
        st.button('USE THIS NEW DATASET', disabled = True, key = -1871878979287)
    
    else:
        
        if st.button('USE THIS NEW DATASET'):
            with st.spinner('Removing outliers manually'):
            
                bad_rows = [int(key) for key in st.session_state['outliers_points']]
                clean = CleanRows(st.session_state['x_df_outliers'], st.session_state['y_df_outliers'])
                clean.remove_rows(bad_rows)
                st.session_state['x_df_outliers'], st.session_state['y_df_outliers'] = clean.x_df, clean.y_df
                
                st.session_state['synthetic_features_created'] = 1
                st.session_state['outliers_removal'] = 2
                st.experimental_rerun()


if st.session_state['outliers_removal'] == 2:
    
    
    selected_date = [st.session_state['selected_points'][x]['date'] for x in st.session_state['selected_points']]
    selected_y = [st.session_state['selected_points'][x]['y'] for x in st.session_state['selected_points']]
    
    outliers_date = [st.session_state['outliers_points'][x]['date'] for x in st.session_state['outliers_points']]
    outliers_y = [st.session_state['outliers_points'][x]['y'] for x in st.session_state['outliers_points']]

    plot = px.scatter()
    plot.add_scatter(x= selected_date, y = selected_y, mode = 'markers', marker = dict(color = 'green'), name = 'points kept')
    plot.add_scatter(x = outliers_date, y = outliers_y, mode = 'markers', marker = dict(color = 'red'), name = 'points removed')
    plot.update_layout(title = {'text' : 'Baseline as a function of time','x':0.47, 'xanchor': 'center', 'yanchor': 'top'},
                       xaxis_title ='Time', yaxis_title='Baseline')
    st.session_state['selected_pt'] = plotly_events(plot, click_event = False, key = st.session_state['iter'])
    


    st.write('')
    st.write('The dataset has been updated succesfully !')    
    
    st.write('Do you want to change the dataset ?  Everything you have done afterwards will be lost.')
    
    if st.button('CHANGE', key = 'outliers_removed'):
        st.session_state['outliers_removal'] = 1
        st.session_state['selected_pt'] = []
        st.session_state['synthetic_features_created'] = 0
        st.session_state['filters_applied'] = 0
        st.session_state['filters_manual'] = 0
        st.session_state['regression_done'] = 0
        st.session_state['results'] = 0
        initialize_results_df_outliers(st)
        initialize_selections(st)
        initialize_selected_outliers_points(st)
        delete_selected(st)
        st.experimental_rerun()
    
    
    st.write('')


    col1, col2, col3 = st.columns([1, 5, 1])        
    
    with col3:
        if st.button("Next >"):
            nav_page("Synthetic_features")
                