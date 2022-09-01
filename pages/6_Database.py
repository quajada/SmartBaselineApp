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
from os.path import exists
import csv
from datetime import datetime
import json
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
    
    if st.button('ADD TO DATABASE', key ='update_database') or ('updating_database' in st.session_state and st.session_state['updating_database']):
        st.session_state['database'] = 1.1
        st.experimental_rerun()
        

if st.session_state['database'] == 1.1:
        
    file_name = 'database.json'       
    
    if not exists(file_name):
        f = open(file_name, 'w')

    f = open(file_name)
    e = st.session_state['excel']
    if os.stat(file_name).st_size == 0:
        db = {}
    else:
        db = json.load(f)
    
    st.session_state['old_db'] = db.copy()
    new_database = e.data.copy()
    st.session_state['new_database'] = new_database
    st.session_state['file_name'] = file_name
    st.session_state['project_name'] = new_database['Project name']
    st.session_state['db'] = db
    st.session_state['database'] = 1.2
    
    st.experimental_rerun()
    
    
    
if st.session_state['database'] == 1.2:
    
    if st.session_state['project_name'] in st.session_state['old_db']:
        st.write('This project name has been already used. You could either change the name of your current project, or replace the old project by the new project and lose the past information.')
        st.session_state['choice'] = st.selectbox(label = 'Pick an option to continue', options = ['Change the name', 'Replace old project'])
    
        if st.button('Final choice ?', key = 198908907316789):
            st.session_state['database'] = 1.3
            st.experimental_rerun()
            
    else:
        st.session_state['database'] = 1.3
        st.experimental_rerun()
    
    
if st.session_state['database'] == 1.3:
    
    if st.session_state['project_name'] in st.session_state['old_db']:
        st.selectbox(label = 'Pick an option to continue', options = [st.session_state['choice']], disabled = True)
        st.button('Final choice ?', key = 87897879, disabled = True)        
        
        if 'Change' in st.session_state['choice']:
            st.session_state['new_name'] = st.text_input('Input the final name')
            
            if not st.button('Confirm name', key = -178678687):
                st.stop()
            else:
                st.session_state['new_database']['Project name'] = st.session_state['new_name']
        else:
            del st.session_state['db'][st.session_state['project_name']]
            
        st.session_state['database'] = 1.4
        st.experimental_rerun()
    
    else:
        st.session_state['database'] = 1.4
        st.experimental_rerun()
    
    

if st.session_state['database'] == 1.4:
    
    with st.spinner('Updating the database'):
        
        new_database = st.session_state['new_database']
        db = st.session_state['db']
        e = st.session_state['excel']
        
        new_database['Start date'] = e.start.strftime('%d/%m/%Y %H:%M')
        new_database['End date'] = e.end.strftime('%d/%m/%Y %H:%M')
        
        new_database['Baseline'] = st.session_state['y_df_regression'].tolist()
        
        sel_combi2 = tuple(st.session_state["final_model"][0]['combinations'])
        sel_version = st.session_state["final_model"][0]['version']
        
        new_database['combination'] = list(sel_combi2)
        
        new_database['r2'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['r2']
        new_database['std_dev'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['std_dev']
        new_database['r2_cv_test'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['r2_cv_test']
        new_database['std_dev_cv_test'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['std_dev_cv_test']
        new_database['cv_rmse'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['cv_rmse']
        new_database['AIC'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['AIC']
        new_database['AIC_adj'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['AIC_adj']
        new_database['tval'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['tval']
        new_database['pval'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['pval']
        new_database['coefficients'] = {}
        new_database['coefficients']['intercept'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['intercept']
        new_database['coefficients']['slopes'] = st.session_state['results_dict'+str(sel_combi2)+str(sel_version)][sel_combi2]['slopes']
        new_database['equation'] = st.session_state['equation']
        
        new_database['created at'] = datetime.now().strftime('%d/%m/%Y %H:%M')
        
        db[new_database['Project name']] = new_database
        
        # os.remove(file_name)
        
        with open(st.session_state['file_name'], 'w') as f:
            json.dump(db, f)
        
        st.session_state['db'] = db
        st.session_state['database'] = 2
        st.experimental_rerun()
    
            
if st.session_state['database'] == 2:
    
    if st.button('OKK', key = 1902890819081908129081290182093):
        st.session_state['database'] = 1
        st.experimental_rerun()
        
    st.write('Data updated succesfully')
    
    if st.checkbox('Show the new database', value = True):
        st.write(st.session_state['db'])
    st.write('')
    st.write('')
    col1, col2, col3 = st.columns([1, 5, 1])        

    with col1:
        if st.button("< Prev"):
            nav_page('Results')