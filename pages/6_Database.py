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
import git
from git import Repo
from git import rmtree
from os.path import exists


initialization(st)

if 'name_sidebar' in st.session_state:
    st.sidebar.title("Project name : " + st.session_state['name_sidebar'])



st.header('Update the database')


if st.session_state['database'] == 0:
    st.write("You'll be able to update the database once you have picked your final model.")


if st.session_state['database'] == 1:
    
    
    st.write('The final model is : **'+ st.session_state['equation']+ '**')
    
    if st.button('ADD "' + st.session_state['name_sidebar'] + '" TO DATABASE', key ='update_database') or ('updating_database' in st.session_state and st.session_state['updating_database']):
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
    
    new_database = e.data.copy()
    st.session_state['new_database'] = new_database
    st.session_state['file_name'] = file_name
    st.session_state['project_name'] = new_database['Project name']
    st.session_state['scope_name'] = new_database['Scope']
    st.session_state['db'] = db
    st.session_state['database'] = 1.2
    
    st.experimental_rerun()
    
    
    
if st.session_state['database'] == 1.2:
    
    if st.session_state['project_name'] in st.session_state['db']:
        
        if st.session_state['scope_name'] in st.session_state['db'][st.session_state['project_name']]: 
        
            if st.session_state['new_database']['Utility']['name'] in [st.session_state['db'][st.session_state['project_name']][st.session_state['scope_name']][i]['Utility']['name'] for i in range (len(st.session_state['db'][st.session_state['project_name']][st.session_state['scope_name']]))] :
                
                st.write('One or more projects with the name ' + st.session_state['name_sidebar']+  ' already exist for the same utility. You could change the name of your current project, or add the new project next to the old project(s), or replace the old project(s) by the new project and lose the past information.')
                st.session_state['choice'] = st.selectbox(label = 'Pick an option to continue', options = ['Change the name', 'Add entry to existing project', 'Replace old project(s)'])
        
                if st.button('Final choice ?', key = 198908907316789):
                    st.session_state['database'] = 1.3
                    st.experimental_rerun()
                    
                else:
                    st.stop()
                    
            else:
                st.session_state['database'] = 1.3
                st.experimental_rerun()
                
        else:
            st.session_state['database'] = 1.3
            st.experimental_rerun()
    
    else:
        st.session_state['database'] = 1.3
        st.experimental_rerun()
  
    
  
if st.session_state['database'] == 1.3:
    
    new_database = st.session_state['new_database']
    db = st.session_state['db']
    e = st.session_state['excel']
    
    # new_database['Start date'] = e.start.strftime('%d/%m/%Y %H:%M')
    # new_database['End date'] = e.end.strftime('%d/%m/%Y %H:%M')

            
        
    sel_combi2 = tuple(st.session_state["final_model"][0]['combinations'])
    sel_version = st.session_state["final_model"][0]['version']
    
    string = str(sel_combi2) + str(sel_version)
    
    new_database['combination'] = {
        'names': list(sel_combi2)
        }
    
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
    
    new_database['number of results'] = st.session_state['nb_of_results']
    new_database['created at'] = datetime.now().strftime('%d/%m/%Y %H:%M')
    
    df = st.session_state['y_df_with_dates']
    
    for feature in sel_combi2:
        df[feature] = st.session_state['x_df_synthetic'][feature]
        
    df = df[df['From (incl)'] >= new_database['Baseline from']]
    df = df[df['To (excl)'] <= new_database['Baseline to']]

    new_database['Baseline'] = {
        'from': list(map(str, df['From (incl)'])),
        'to': list(map(str, df['To (excl)'])),
        'baseline': df['Baseline'].tolist(),
        # 'normalized baseline': df['Normalized baseline'].tolist()
        }
    
    for feature in sel_combi2:
        new_database['combination'][feature] = df[feature].tolist()
    
    new_database['Baseline from'] = str(new_database['Baseline from'])
    new_database['Baseline to'] = str(new_database['Baseline to'])
    
    
    from_list = [st.session_state['y_df_results']['From (incl)'][time] for time in st.session_state['selected_points'+string]]
    to = [st.session_state['y_df_results']['To (excl)'][time] for time in st.session_state['selected_points'+string]]
    
    timedelta = np.zeros(len(to))
    for i in range(len(timedelta)):
        timedelta[i] = (to[i]-from_list[i]).total_seconds()/3600
    
    new_database['Target'] = {
        'from': list(map(str, from_list)),
        'to': list(map(str, to)),
        'timedelta': timedelta.tolist(),
        'normalized target':  [st.session_state['y_df_results']['Normalized baseline'][time] for time in st.session_state['selected_points'+string]],
        'target':  [st.session_state['y_df_results']['Baseline'][time] for time in st.session_state['selected_points'+string]]
        }
    
    y_pred = st.session_state['results_dict'+string][sel_combi2]['y_pred']
    y_test = new_database['Target']['normalized target']
    y = (y_test - y_pred)
    timedelta = new_database['Target']['timedelta']
    for k in range(len(y)):
        y[k] = y[k] * timedelta[k]
    y = y**2
    std_dev = np.sqrt( sum(y) / (len(y)-len(sel_combi2)-1) )
    new_database['std_dev'] = std_dev
    
    
    for feature in sel_combi2:
        new_database['combination'][feature] = [st.session_state['x_df_results'][feature][time] for time in st.session_state['selected_points'+string]]

    st.session_state['database'] = 1.4
    st.experimental_rerun()
    

if st.session_state['database'] == 1.4:
    
    if st.session_state['project_name'] in st.session_state['db']:
        
        if st.session_state['scope_name'] not in st.session_state['db'][st.session_state['project_name']]:
            
            st.session_state['db'][st.session_state['project_name']][st.session_state['scope_name']] = [st.session_state['new_database']]

        else:
                        
            if st.session_state['new_database']['Utility']['name'] in [st.session_state['db'][st.session_state['project_name']][st.session_state['scope_name']][i]['Utility']['name'] for i in range (len(st.session_state['db'][st.session_state['project_name']][st.session_state['scope_name']]))] :
    
                st.selectbox(label = 'Pick an option to continue', options = [st.session_state['choice']], disabled = True)
                st.button('Final choice ?', key = 87897879, disabled = True)        
            
                if 'Change' in st.session_state['choice']:
                    st.session_state['new_name'] = st.text_input('Input the final name')
                
                    if not st.button('Confirm name', key = -178678687):
                        st.stop()
                    else:
                        if st.session_state['new_name'] in st.session_state['db']:
                            st.write('**This name is already in the database. Please choose another name.**')
                            st.stop()
                        else:
                            st.session_state['new_database']['Project name'] = st.session_state['new_name']
                            st.session_state['db'][st.session_state['new_name']] = {}
                            st.session_state['db'][st.session_state['new_name']][st.session_state['scope_name']] = [st.session_state['new_database']]
            
                elif 'Add' in st.session_state['choice']:
                    st.session_state['db'][st.session_state['project_name']][st.session_state['scope_name']].append(st.session_state['new_database'])
                    
                else:
                    bad_indexes = []
                    for i in range (len(st.session_state['db'][st.session_state['project_name']][st.session_state['scope_name']])):
                        if st.session_state['db'][st.session_state['project_name']][st.session_state['scope_name']][i]['Utility']['name'] == st.session_state['new_database']['Utility']['name']:
                            bad_indexes.append(i)
                            
                    for index in sorted(bad_indexes, reverse= True):
                        del st.session_state['db'][st.session_state['project_name']][st.session_state['scope_name']][index]
    
                    st.session_state['db'][st.session_state['project_name']][st.session_state['scope_name']].append(st.session_state['new_database'])
                    
            else:
                st.session_state['db'][st.session_state['project_name']][st.session_state['scope_name']].append(st.session_state['new_database'])
                st.session_state['database'] = 1.5
                st.experimental_rerun()
            
        st.session_state['database'] = 1.5
        st.experimental_rerun()
        
    else:
        st.session_state['db'][st.session_state['project_name']] = {}
        st.session_state['db'][st.session_state['project_name']][st.session_state['scope_name']] = [st.session_state['new_database']]
        st.session_state['database'] = 1.5
        st.experimental_rerun()
    
    

if st.session_state['database'] == 1.5:
    
    with st.spinner('Updating the database'):
        
        if exists('test'):
            rmtree('test')
        
        # repo = Repo.clone_from('https://BrunoTabet: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@github.com/BrunoTabet/SmartBaseline.git', "test")
        repo = Repo.clone_from('https://BrunoTabet:02Mars2001Tokyo@github.com/BrunoTabet/SmartBaseline.git', "test")
        # repo = Repo.clone_from('git@github.com:BrunoTabet/Smart-Baseline.git', "test")
        repo.remote().fetch()
        
        with open(st.session_state['file_name'], 'w') as f:
            json.dump(st.session_state['db'], f)
            
        with open('test/database.json', 'w') as f:
            json.dump(st.session_state['db'], f)
        
        # repo = Repo("test")
        repo.git.add(update=True)
        # repo.git.add(['database.json'])
        repo.index.commit("latest commit from Streamlit")
        origin = repo.remote(name='origin')
        origin.push()
        repo.close()
        
        rmtree('test')
    
        st.session_state['database'] = 2
        st.session_state['M&V'] = 1
        st.experimental_rerun()
        
            
if st.session_state['database'] == 2:
        
    st.write('Data updated successfully !')

    
    if st.checkbox('Show the new database', value = False):
        st.write(st.session_state['db'])
    st.write('')
    st.write('')
    col1, col2, col3 = st.columns([1, 5, 1])        

    with col1:
        if st.button("< Prev"):
            nav_page('Results')
    
    with col3:
        if st.button("Next >"):
            nav_page('M&V_report')
            