# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 12:06:26 2022

@author: Bruno Tabet
"""


import streamlit as st
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import plotly_express as px
import openpyxl
import xlsxwriter as xl
import xlwt
import xlrd
from xlutils.copy import copy
import os
import csv

# sys.path.append(r"C:\Users\Bruno Tabet\Documents\ENOVA\MVP")


from engines.engine import Engine
from helpful_funcs.excel_funcs import ReadExcel
from combinations.combinations import Combinations
from helpful_funcs.useful_funcs import nav_page
from helpful_funcs.useful_funcs import initialization


initialization(st)



st.header('Update the database')

if st.session_state['database'] == 0:
    st.write("You'll be able to update the database once you have picked your final model.")


if st.session_state['database'] == 1:
    
    st.write('The final model is : **'+ st.session_state['equation']+ '**')
    
    if st.button('ADD TO DATABASE', key ='update database'):
        
        e = st.session_state['excel']
    
        database = e.data
        database['Start date'] = e.start
        database['End date'] = e.end

        database['Baseline'] = st.session_state['y_df_regression'].tolist()
        
        sel_combi2 = tuple(st.session_state["final_model"][0]['combinations'])
        sel_version = st.session_state["final_model"][0]['version']
        
        database['combination'] = list(sel_combi2)
        
        database['r2'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['r2']
        database['std_dev'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['std_dev']
        database['r2_cv_test'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['r2_cv_test']
        database['std_dev_cv_test'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['std_dev_cv_test']
        database['cv_rmse'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['cv_rmse']
        database['AIC'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['AIC']
        database['AIC_adj'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['AIC_adj']
        database['tval'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['tval']
        database['pval'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['pval']
        database['coefficients'] = {}
        database['coefficients']['intercept'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['intercept']
        database['coefficients']['slopes'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['slopes']
        database['equation'] = st.session_state['equation']
    
        st.session_state['new_database'] = database
        st.session_state['database'] = 2
        st.experimental_rerun()
    
            
if st.session_state['database'] == 2:
    
    st.write('Data updated succesfully')
    
    if st.checkbox('Show the new database', value = True):
        st.write(st.session_state['new_database'])
    st.write('')
    st.write('')
    col1, col2, col3 = st.columns([1, 5, 1])        

    with col1:
        if st.button("< Prev"):
            nav_page('Results')