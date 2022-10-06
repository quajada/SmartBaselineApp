# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:04:46 2022

@author: Bruno Tabet
"""


import streamlit as st
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import plotly_express as px
import openpyxl
import statsmodels.api as sm
from statistics import mean, stdev, median


# sys.path.append(r"C:\Users\Bruno Tabet\Documents\ENOVA\MVP")

from engines.engine import Engine
from helpful_funcs.excel_funcs import ReadExcel
from combinations.combinations import Combinations
from streamlit_plotly_events import plotly_events
from features.synthetic_features import SyntheticFeatures, CDD, HDD
from cleandata.cleandata import Aggregator, CleanColumns, CleanRows
from helpful_funcs.excel_funcs import ReadExcel
from combinations.combinations import Combinations
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from helpful_funcs.useful_funcs import *

initialization(st)

if 'name_sidebar' in st.session_state:
    st.sidebar.title("Project name : " + st.session_state['name_sidebar'])


st.header('Results')

if st.session_state['results'] == 0:
    
    st.write('The regression is not finished yet.')


if st.session_state['results'] == 1:
    
    st.session_state['x_df_results'] = st.session_state['x_df_regression'].copy()
    st.session_state['y_df_results'] = st.session_state['y_df_regression'].copy()
    
    
    st.write("All the models are shown in the table below. Select a model to get more information.")
    st.write("The main statistical parameters of the regression are shown. **_r2_cv_test_** and **_std_dev_cv_test_** correspond to the average of the **_r2_** and **_std_dev_** on the test data of all the scenarios in the cross validation.")
    
    # st.write(st.session_state['results_df'].dtypes)
    
    gd = GridOptionsBuilder.from_dataframe(st.session_state['results_df'])
    gd.configure_selection(selection_mode='single',use_checkbox=True)
    # gd.configure_default_column()
    # gd.configure_column('r2', columnSize = 'SizeToFit', width = 10)
    gridoptions = gd.build()
    grid1 = AgGrid(st.session_state['results_df'],gridOptions=gridoptions,
                        update_mode= GridUpdateMode.SELECTION_CHANGED)
    
    st.write(str(len(st.session_state['results_df'])) + " rows, " + str(len(st.session_state['results_df'].columns))+ " columns")
    st.session_state['nb_of_results'] = len(st.session_state['results_df'])
    
    sel_row = grid1["selected_rows"]
    
    if len(sel_row) > 0:
        
        sel_combi = tuple(sel_row[0]['combinations'])
        
        if 'iteration'+str(sel_combi) not in st.session_state:
            st.session_state['iteration' + str(sel_combi)] = 1
        
        itera = st.session_state['iteration' + str(sel_combi)]
        
        string = str(sel_combi) + str(itera)
        
        if 'results_dict'+str(sel_combi)+str(itera) not in st.session_state:                
            st.session_state['results_dict'+str(sel_combi)+str(itera)] = st.session_state['results_dict']
            
        if 'results_df'+str(sel_combi)+str(itera) not in st.session_state:
            st.session_state['results_df'+str(sel_combi)+str(itera)] = st.session_state['results_df']
            
        if 'x_df'+str(sel_combi)+str(itera) not in st.session_state:
            st.session_state['x_df'+str(sel_combi)+str(itera)] = st.session_state['x_df_results']
            
        if 'y_df'+str(sel_combi)+str(itera) not in st.session_state:
            st.session_state['y_df'+str(sel_combi)+str(itera)] = st.session_state['y_df_results']
        
        if 'selected_points' + string not in st.session_state and 'outliers_points' + string not in st.session_state:
            
            st.session_state['selected_points'+string] = {}
            st.session_state['outliers_points'+string] = {}
            
            y_pred_df = pd.DataFrame(index = st.session_state['y_df'+string].index )
            y_pred_df['y_pred'] = st.session_state['results_dict'+string][sel_combi]['y_pred']
            
            
            for time in st.session_state['y_df'+string].index:
                st.session_state['selected_points'+string][time]= {}
                st.session_state['selected_points'+string][time]['baseline'] = st.session_state['y_df'+string]['Normalized baseline'][time]
                st.session_state['selected_points'+string][time]['prediction'] = y_pred_df['y_pred'][time]

            
        selected_baseline = [st.session_state['selected_points'+string][time]['baseline'] for time in st.session_state['selected_points'+string]]
        selected_prediction = [st.session_state['selected_points'+string][time]['prediction'] for time in st.session_state['selected_points'+string]]
        
        outliers_baseline = [st.session_state['outliers_points'+string][time]['baseline'] for time in st.session_state['outliers_points'+string]]
        outliers_prediction = [st.session_state['outliers_points'+string][time]['prediction'] for time in st.session_state['outliers_points'+string]]
        
        plot = px.scatter()
        plot.add_scatter(x= selected_baseline, y = selected_prediction, mode = 'markers', marker = dict(color = 'blue'), name = 'points to keep')
        plot.add_scatter(x = outliers_baseline, y = outliers_prediction, mode = 'markers', marker = dict(color = 'red'), name = 'points to remove')
        plot.update_layout(title = {'text' : 'Predictions of '+ str(sel_combi)+ ' as a function of the baseline','x':0.47, 'xanchor': 'center', 'yanchor': 'top'},
                           xaxis_title ='Baseline', yaxis_title='Predictions')
        plot.add_scatter(x = st.session_state['y_df'+str(sel_combi)+str(itera)]['Normalized baseline'], y = st.session_state['y_df'+str(sel_combi)+str(itera)]['Normalized baseline'], mode='lines', marker = dict(color = 'green'), name = 'y = x')
        
        # plot = px.scatter( x = st.session_state['y_df'+str(sel_combi)+str(itera)], y = st.session_state['results_dict'+str(sel_combi)+str(itera)][sel_combi]['y_pred'], labels = {'x':'Baseline', 'y':'optimized predictions for '+ str(sel_combi) + str(itera)})
        
        y = st.session_state['y_df' + str(sel_combi)+str(itera)]['Normalized baseline']
        std_dev = st.session_state['results_dict'][sel_combi]['std_dev']
        
        colours = ['pink', 'yellow', 'orange']
        
        for i in range (0,3):
            plot.add_scatter(x = y, y = y- (i+1)*std_dev, mode = 'lines', line = dict(shape = 'linear', color = colours[i], width= 0.5), name = str(i+1)+' std_dev')
            plot.add_scatter(x = y, y = y + (i+1)*std_dev, mode = 'lines', line = dict(shape = 'linear', color = colours[i], width= 0.5), name = '')
            
        st.session_state['selected_pt'+str(sel_combi)+str(itera)] = plotly_events(plot, click_event=True, key = st.session_state['iter'])



        st.write("You can optimize this model by removing outliers. The optimized model will then appear on the second table below. Click on a point to remove/add it.")
        
        if st.session_state['selected_pt'+str(sel_combi)+str(itera)] == []:
            if 'time'+string in st.session_state:
                st.write('The point selected corresponds to time = '+str(st.session_state['time'+string]))
        
        else:
            baseline = st.session_state['selected_pt'+str(sel_combi)+str(itera)][0]['x']
            prediction = st.session_state['selected_pt'+str(sel_combi)+str(itera)][0]['y']
            
            time = None
            for t in st.session_state['selected_points'+string]:
                if st.session_state['selected_points'+string][t]['baseline'] == float(baseline) and st.session_state['selected_points'+string][t]['prediction'] == float(prediction):
                    time = t
                    
            for t in st.session_state['outliers_points'+string]:
                if st.session_state['outliers_points'+string][t]['baseline'] == float(baseline) and st.session_state['outliers_points'+string][t]['prediction'] == float(prediction):
                    time = t
            
            if time == None:
                st.write("You haven't selected a blue point.")
                        
            else:
                if time in st.session_state['selected_points' + string]:
                    
                    if len(st.session_state['outliers_points'+string]) >= st.session_state['test_size'] + 1:
                        st.write("You can't remove any more points because the test size is " + str(st.session_state['test_size']) + ". The total number of data points you have has to be greater than your test size.")
                        st.session_state['selected_pt'+str(sel_combi)+str(itera)] = []
                        st.session_state['iter']+=1
                        st.stop()
                    
                    else:
                        del st.session_state['selected_points'+string][time]
                        st.session_state['outliers_points'+string][time] = {}
                        st.session_state['outliers_points'+string][time]['baseline'] = baseline
                        st.session_state['outliers_points'+string][time]['prediction'] = prediction
                        
                
                else:
                    del st.session_state['outliers_points'+string][time]
                    st.session_state['selected_points'+string][time] = {}
                    st.session_state['selected_points'+string][time]['baseline'] = baseline
                    st.session_state['selected_points'+string][time]['prediction'] = prediction
                
                st.session_state['time'+string] = time
                st.session_state['iter'] += 1
                st.experimental_rerun()
                    
        
        # st.write('Are these the final outliers you want to remove for your model optimization ?')
        col1, col2 = st.columns([8, 2])
        
        if len(st.session_state['selected_points'+string]) <= st.session_state['test_size']:
            st.write('')
            col1.write('**You need to select more points than your test size which is '+ str(st.session_state["test_size"])+ 
                     '. You are currently selecting only '+ str(len(st.session_state['selected_points'+string])) +' data points.**')
            col2.button('ADD MODEL TO FINAL SELECTION', disabled = True)
            
            
        else:
            if col2.button('ADD MODEL TO FINAL SELECTION', key = sel_combi):
                with st.spinner('Removing outliers and computing the optimized model'):
                    
                    drapeau = False
                    for i in st.session_state['results_df_outliers'].index:
                        if st.session_state['results_df_outliers']['combinations'][i] == sel_combi:
                            if st.session_state['results_df_outliers']['version'][i] == 0:
                                drapeau = True
                    
                    
                    if drapeau == False:
                        new_dict = {}
                        new_dict['combinations'] = sel_combi
                        columns_list = ['pval', 'tval']
                        
                        for column in st.session_state['results_df_outliers']:
                            
                            if column in columns_list:
                                round_values = st.session_state['results_dict'][sel_combi][column].copy()
                                for k in range (len(round_values)):
                                    round_values[k] = round(round_values[k], 4)
                                new_dict[column] = round_values
                            
                            else:
                                if column not in ['combinations','version', 'nb_data_points', 'nb_outliers_removed']:
                                    new_dict[column] = st.session_state['results_dict'][sel_combi][column]
                            
                            new_dict['nb_outliers_removed'] = 0
                            new_dict['version'] = 0
                            new_dict['nb_data_points'] = len(st.session_state['x_df_results']) - new_dict['nb_outliers_removed']
                        
                        columns_float = ['r2', 'std_dev', 'r2_cv_test', 'std_dev_cv_test', 'intercept', 'AIC', 'AIC_adj']
                        for column in columns_float:
                            new_dict[column] = round(new_dict[column], 3)
                        
                        st.session_state['results_df_outliers'] = st.session_state['results_df_outliers'].append(new_dict, ignore_index = True)
                    
                    is_already = False
                    
                    if 'outliers_points'+string not in st.session_state:
                        is_already = True
                        
                    elif st.session_state['outliers_points'+string] == {}:
                        is_already = True
                    
                    else:
                        for i in range (1, st.session_state['iteration'+str(sel_combi)]):
                            if st.session_state['outliers_points'+str(sel_combi)+str(i)] == st.session_state['outliers_points'+string]:
                                is_already = True
                            
                    # st.session_state['is_already'+str(sel_combi)] = is_already
                    if is_already:
                        col1.write('**The model you chose is already in the final selection**')
                    
                    else:
                        times = [time for time in st.session_state['outliers_points'+string]]
                        
                        clean = CleanRows(st.session_state['x_df'+str(sel_combi)+str(itera)], st.session_state['y_df'+str(sel_combi)+str(itera)])
                        clean.remove_rows(times)
                        st.session_state['x_df'+str(sel_combi)+str(itera)], st.session_state['y_df'+str(sel_combi)+str(itera)] = clean.x_df, clean.y_df
                        
                        final2 = Engine(st.session_state['x_df'+str(sel_combi)+str(itera)], st.session_state['y_df'+str(sel_combi)+str(itera)], [sel_combi], max_variables = st.session_state['max_features'], nb_folds = st.session_state['nb_folds'], test_size = st.session_state['test_size'])
                        final2.compute_cross_validation()
                        final2.are_combinations_IPMVP_compliant()
                        final2.get_df_results()
                        st.session_state['results_df'+str(sel_combi)+str(itera)] = final2.results_df
                        st.session_state['results_dict'+str(sel_combi)+str(itera)] = final2.results
                        
                        new_dict = {}
                        new_dict['combinations'] = sel_combi
                        for column in st.session_state['results_df_outliers']:
                            if column not in ['combinations','version', 'nb_data_points', 'nb_outliers_removed']:
                                new_dict[column] = st.session_state['results_df'+str(sel_combi)+str(itera)][column]
                       
                        new_dict['nb_outliers_removed'] = len(st.session_state['outliers_points'+string])
                        new_dict['version'] = itera
                        new_dict['nb_data_points'] = len(st.session_state['selected_points'+string])
                        
                        st.session_state['results_df_outliers'] = st.session_state['results_df_outliers'].append(new_dict, ignore_index = True)
            
                        st.session_state['iteration' + str(sel_combi)] += 1
                        st.session_state['selected_pt'+str(sel_combi)+str(itera)] = {}
                        st.experimental_rerun()                    
                
    st.write('')
    st.write('')
    
    st.write("The final and optimized model selection is shown below. A model can have multiple versions corresponding to the outliers removed. For version 0 we kept the entire dataset (this version is automatically created)."+
             " Select an optimized model to get more information.")
    

    columns_float = ['nb_data_points', 'nb_outliers_removed', 'version', 'r2', 'std_dev', 'r2_cv_test', 'std_dev_cv_test', 'intercept', 'cv_rmse', 'AIC', 'AIC_adj', 'size']
    for column in columns_float:
        st.session_state['results_df_outliers'][column] = st.session_state['results_df_outliers'][column].astype(np.float64)
    
    columns_int =  ['nb_data_points', 'nb_outliers_removed', 'version']
    for column in columns_int:
        st.session_state['results_df_outliers'][column] = pd.to_numeric(st.session_state['results_df_outliers'][column])
    
    gd = GridOptionsBuilder.from_dataframe(st.session_state['results_df_outliers'])
    gd.configure_selection(selection_mode='single',use_checkbox=True)
    gd.configure_default_column(editable=True, groupable=True)
    gridoptions = gd.build()
    grid2 = AgGrid(st.session_state['results_df_outliers'],gridOptions=gridoptions,
                        update_mode= GridUpdateMode.SELECTION_CHANGED)
    
    st.write(str(len(st.session_state['results_df_outliers'])) + " rows, " + str(len(st.session_state['results_df_outliers'].columns))+ " columns")

    
    sel_row2 = grid2["selected_rows"]

    col_3, col_4 = st.columns(2)
    
    if len(sel_row2) > 0:
        sel_combi2 = tuple(sel_row2[0]['combinations'])
        sel_version = sel_row2[0]['version']
        string = str(sel_combi2)+str(sel_version)
        
        if sel_version == 0:
            st.session_state['results_dict'+str(sel_combi2)+str(sel_version)] = st.session_state['results_dict']
            st.session_state['results_df'+str(sel_combi2)+str(sel_version)] = st.session_state['results_df']
            st.session_state['x_df'+str(sel_combi2)+str(sel_version)] = st.session_state['x_df_results']
            st.session_state['y_df'+str(sel_combi2)+str(sel_version)] = st.session_state['y_df_results']
            
            st.session_state['selected_points'+string] = {}
            st.session_state['outliers_points'+string] = {}
            
            y_pred_df = pd.DataFrame(index = st.session_state['y_df'+string].index )
            y_pred_df['y_pred'] = st.session_state['results_dict'+string][sel_combi2]['y_pred']
            
            for time in st.session_state['y_df'+string].index:
                st.session_state['selected_points'+string][time]= {}
                st.session_state['selected_points'+string][time]['baseline'] = st.session_state['y_df'+string]['Normalized baseline'][time]
                st.session_state['selected_points'+string][time]['prediction'] = y_pred_df['y_pred'][time]

        
        with col_3:
            
            selected_baseline = [st.session_state['selected_points'+string][time]['baseline'] for time in st.session_state['selected_points'+string]]
            selected_prediction = [st.session_state['selected_points'+string][time]['prediction'] for time in st.session_state['selected_points'+string]]
            
            outliers_baseline = [st.session_state['outliers_points'+string][time]['baseline'] for time in st.session_state['outliers_points'+string]]
            outliers_prediction = [st.session_state['outliers_points'+string][time]['prediction'] for time in st.session_state['outliers_points'+string]]
            

            plot = px.scatter(labels = {'x':'Time', 'y':'Baseline'})
            plot.add_scatter(x= selected_baseline, y = selected_prediction, mode = 'markers', marker = dict(color = 'blue'), name = 'points kept')
            plot.add_scatter(x = outliers_baseline, y = outliers_prediction, mode = 'markers', marker = dict(color = 'red'), name = 'points removed')
            
            plot.add_scatter(x = st.session_state['y_df'+string]['Normalized baseline'], y = st.session_state['y_df'+string]['Normalized baseline'], mode='lines', marker = dict(color = 'green'), name = 'y = x')
            plot.update_layout(title = {'text' : 'Predictions of '+ str(sel_combi2) +' as a function of the baseline','x':0.47, 'xanchor': 'center', 'yanchor': 'top'},
                               xaxis_title ='Baseline', yaxis_title='Predictions')
            plotly_events(plot, click_event=False, key = -st.session_state['iter']-1)
            
            
            if st.button('THIS IS MY FINAL MODEL', key = 10000):
                st.session_state["final_model"] = sel_row2
                st.session_state['results'] = 2
                st.session_state['database'] = 1
                nav_page("Database")
            
        with col_4:            
            display_results(sel_combi2, sel_version, st)

      
    st.write('')
    st.write('')
 


    
if st.session_state['results'] == 2:
    
    sel_combi2 = tuple(st.session_state["final_model"][0]['combinations'])
    sel_version = st.session_state["final_model"][0]['version']
    
    string = str(sel_combi2) + str(sel_version)
    
    st.subheader('This is your final model : ' + str(sel_combi2))
 
    selected_baseline = [st.session_state['selected_points'+string][time]['baseline'] for time in st.session_state['selected_points'+string]]
    selected_prediction = [st.session_state['selected_points'+string][time]['prediction'] for time in st.session_state['selected_points'+string]]
    
    outliers_baseline = [st.session_state['outliers_points'+string][time]['baseline'] for time in st.session_state['outliers_points'+string]]
    outliers_prediction = [st.session_state['outliers_points'+string][time]['prediction'] for time in st.session_state['outliers_points'+string]]
    
    col_3, col_4 = st.columns(2)
    
    with col_3:

        plot = px.scatter(labels = {'x':'Time', 'y':'Baseline'})
        plot.add_scatter(x= selected_baseline, y = selected_prediction, mode = 'markers', marker = dict(color = 'blue'), name = 'points kept')
        plot.add_scatter(x = outliers_baseline, y = outliers_prediction, mode = 'markers', marker = dict(color = 'red'), name = 'points removed')
        plot.add_scatter(x = st.session_state['y_df_results']['Normalized baseline'], y = st.session_state['y_df_results']['Normalized baseline'], mode='lines', marker = dict(color = 'green'), name = 'y = x')
        plot.update_layout(title = {'text' : 'Predictions as a function of the baseline','x':0.47, 'xanchor': 'center', 'yanchor': 'top'},
                           xaxis_title ='Baseline', yaxis_title='Predictions')
        plotly_events(plot, click_event=False, key = -st.session_state['iter'])
        
        st.write('')
        st.write('')

        st.write('Do you want to change the final model chosen ? Everything you have done afterwards will be lost.')
        
        if st.button('CHANGE'):
            st.session_state['results'] = 1
            st.session_state['database'] = 0
            st.experimental_rerun()

        
    with col_4:
        
        display_results(sel_combi2, sel_version, st)


    st.write('')
    st.write('')   
    st.write('')
    st.write('')
    
    col1, col2, col3 = st.columns([1, 5, 1]) 

    with col1:
        if st.button("< Prev"):
            nav_page('Regression')

    with col3:
        if st.button("Next >"):
            nav_page("Database")
                
