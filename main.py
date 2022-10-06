# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:27:30 2022

@author: Bruno Tabet
"""

import numpy as np
import pandas as pd

import openpyxl

from features.synthetic_features import SyntheticFeatures, CDD, HDD
from cleandata.cleandata import Aggregator, CleanColumns, CleanRows
from filters.filterdata import FilterData
from engines.engine import Engine
from helpful_funcs.excel_funcs import ReadExcel
from combinations.combinations import Combinations
import pickle

if __name__ == "__main__":
    
    # from units import units
    
    # for i in range (15, 31):
    #     units['CDD'+str(i)] = units['CDD'].replace('days', 'days '+str(i))
    #     units['HDD'+str(i)] = units['HDD'].replace('days', 'days '+str(i))
        
    # functions1 = ['max', 'min', 'std', 'sum', 'mean', 'most']
    # functions1_names = ['max of ', 'min of ', 'std of ', 'sum of ', 'mean of ', 'most frequent ']
    
    # units1 = units.copy()

    # for variable in units:
    #     print(variable)
    #     for i in range(len(functions1)):
    #         units1[functions1[i] + '_'+ variable] = functions1_names[i] + units[variable]
    
    
    # functions2 = ['sqrt', 'squared', 'inv']
    # functions2_names = ['sqrt of ', 'square of ', 'inverse of ']
    
    # units2 = units1.copy()
    
    # for variable in units1:
    #     for i in range(len(functions2)):
    #         units2[functions2[i] + '_'+ variable] = functions2_names[i]+ units1[variable]
    
    # print(units2)

    path = r"C:\Users\Bruno Tabet\Documents\ENOVA\MVP\Input_template - Test.xlsx"
    # path = r"C:\Users\Bruno Tabet\Documents\ENOVA\MVP\Input_template.xlsx"
    # path = r"C:\Users\Bruno Tabet\Documents\ENOVA\MVP\Input_template - Turkey.xlsx"
    e = ReadExcel(path)
    
    df_weather, baseline = e.preprocess_data(path)
    x_df, y_df, baseline = e.ok(df_weather, baseline)

    clean = CleanColumns(x_df)
    clean.remove_nan()
    clean.fill_nan()
    clean.remove_duplicates()
    x_df = clean.x_df
    
    # y_df = y_df_dates['Normalized baseline'].astype(np.float64)
    
    start = y_df['From (incl)'].values[0]
    end = y_df['To (excl)'].values[-1]
    
    # clean = CleanRows(x_df, y_df)
    # bad_rows = clean.get_outliers_quantile(quantile = 0.01)
    # clean.remove_rows(bad_rows)
    
    # bad_rows = clean.get_outliers_zvalue(n_std=2)
    # clean.remove_rows(bad_rows)
    # x_df = clean.x_df
    # y_df = clean.y_df
    
    synth = SyntheticFeatures(x_df)
    synth.create_inverse([])
    # synth.create_products()
    synth.create_sqrt_and_squared()
    synth.remove_str_beginning('min')
    x_df = synth.x_df
    
    print(x_df.columns)
    print(len(x_df.columns))
    
    filt = FilterData(x_df, y_df['Normalized baseline'])
    print('')
    
    
    bad_features_pearson = filt.get_bad_features_pearson(0)
    print('THE FEATURES REMOVED BY PEARSON ARE')
    print(bad_features_pearson)
    print(len(bad_features_pearson))
    filt.remove_features(bad_features_pearson)
    
    print('')
    
    bad_features_spearman = filt.get_bad_features_spearman(0)
    print('THE FEATURES REMOVED BY SPEARMAN ARE')
    print(bad_features_spearman)
    print(len(bad_features_spearman))
    filt.remove_features(bad_features_spearman)
    
    print('')
    
    bad_features_info = filt.get_bad_features_info(0)
    print('THE FEATURES REMOVED BY MUTUAL INFORMATION ARE')
    print(bad_features_info)
    print(len(bad_features_info))
    filt.remove_features(bad_features_info)
    
    print('')
    
    bad_other_features = filt.get_worst_features()
    print('THE FINAL FEATURES REMOVED BY PEARSON ARE')
    print(bad_other_features)
    print(len(bad_other_features))
    # filt.remove_features(bad_other_features)
    x_df = filt.x_df
    
    print('')
    print('THE COLUMNS THAT REMAIN AFTER FILTER')
    print(x_df.columns)
    print(len(x_df.columns))
    print('')
    
    combi = Combinations(x_df.columns, 1)
    combinations = combi.compute_combinations(x_df)
    # # combinations = [('passengers',)]
    final = Engine(x_df, y_df, combinations, max_variables = 1, nb_folds = 100, test_size = 18)
    # IPMVP_combinations, IMPVP_combi_compliant = final.get_IPMVP_combinations()
    # # IPMVP_results, IPMVP_compliant = final.compute_IPMVP_results()
    # # not_overfitting_results, no_overfit = final.remove_overfitting()
    # # final_results = final.show_top_results(criteria='std_dev', nb_variables=2, nb_top = 30)
           
    final.compute_cross_validation()
    # final.are_combinations_IPMVP_consistently_compliant()
    final.are_combinations_IPMVP_compliant()
    results_df = final.get_df_results()
    results_dict = final.results