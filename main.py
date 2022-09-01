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


if __name__ == "__main__":
    
    path = r"C:\Users\Bruno Tabet\Documents\ENOVA\Input_template_new_database.xlsx"
    e = ReadExcel(path)
    x_df, y_df_dates = e.preprocess_data(path)
    y_df = y_df_dates['Normalized baseline'].astype(np.float64)
    """
    clean = CleanRows(x_df, y_df)
    bad_rows = clean.get_outliers_quantile(quantile = 0.01)
    clean.remove_rows(bad_rows)
    
    bad_rows = clean.get_outliers_zvalue(n_std=2)
    clean.remove_rows(bad_rows)
    x_df = clean.x_df
    y_df = clean.y_df
    
    synth = SyntheticFeatures(x_df)
    synth.create_inverse([])
    # synth.create_products()
    synth.create_sqrt_and_squared()
    synth.remove_str_beginning('min')
    x_df = synth.x_df
    
    
    print(x_df.columns)
    print(len(x_df.columns))
    
    filt = FilterData(x_df, y_df)
    print('')
    
    bad_features_pearson = filt.get_bad_features_pearson(0.0)
    print('THE FEATURES REMOVED BY PEARSON ARE')
    print(bad_features_pearson)
    print(len(bad_features_pearson))
    filt.remove_features(bad_features_pearson)
    
    print('')
    
    bad_features_spearman = filt.get_bad_features_spearman(0.0)
    print('THE FEATURES REMOVED BY SPEARMAN ARE')
    print(bad_features_spearman)
    print(len(bad_features_spearman))
    filt.remove_features(bad_features_spearman)
    
    print('')
    
    bad_features_info = filt.get_bad_features_info(0.0)
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
    
    
    combi = Combinations(x_df.columns, 2)
    combinations = combi.compute_combinations(x_df)
    # combinations = [('passengers',)]
    final = Engine(x_df, y_df, combinations, max_variables = 2, nb_folds = 100, test_size = 4)
    # IPMVP_combinations, IMPVP_combi_compliant = final.get_IPMVP_combinations()
    # IPMVP_results, IPMVP_compliant = final.compute_IPMVP_results()
    # not_overfitting_results, no_overfit = final.remove_overfitting()
    # final_results = final.show_top_results(criteria='std_dev', nb_variables=2, nb_top = 30)
    
    final.compute_cross_validation()
    # final.are_combinations_IPMVP_consistently_compliant()
    final.are_combinations_IPMVP_compliant()
    results_df = final.get_df_results()  
    results_dict = final.results
    """