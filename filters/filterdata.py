# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 12:11:01 2022

@author: Bruno Tabet
"""

import numpy as np
import statistics
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNet
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression


class FilterData:
    
    def __init__(self, x_df, y_values):
        self.x_df = x_df
        self.y_values = y_values
    
    def split_data(self, testsize, randomstate):
        X_train = self.x_df.values
        y_train = self.y_values
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=testsize, random_state=randomstate)
        return X_train, X_test, y_train, y_test
    
    def remove_features(self, list_bad_features):
        self.x_df = self.x_df.drop([feature for feature in list_bad_features], axis = 1)
        return self.x_df
        
    def intersection_features(self, list_of_lists):
        set_list = []
        for list_features in list_of_lists:
            set_features = set(list_features)
            set_list.append(set_features)
        intersection_set = set.intersection(*set_list)
        return(list(intersection_set))
    
    def union_features(self, list_of_lists):
        set_list = []
        for list_features in list_of_lists:
            set_features = set(list_features)
            set_list.append(set_features)
        union_set = set.union(*set_list)
        return(list(union_set)) 

    
    def get_bad_features_pearson(self, coef_pearson):
        fs = []
        bad_features_pearson = []
        
        print(self.x_df.columns)
        
        for column in self.x_df.columns:
            corr, _ = pearsonr(self.x_df[column], self.y_values)
            fs.append(abs(corr)) 
        
        # mean = statistics.mean(fss)
        # threshold = mean*coef_pearson
        
        for n in range (len(self.x_df.columns)):
            if fs[n] < coef_pearson:
                bad_features_pearson.append(self.x_df.columns[n])

        return bad_features_pearson
    
    

    def get_bad_features_spearman(self, coef_spearman):

        fs = []
        bad_features_spearman = []
        
        for column in self.x_df.columns:
            corr, _ = spearmanr(self.x_df[column], self.y_values)
            fs.append(abs(corr)) 
        
        # mean = statistics.mean(fss)
        # threshold = mean*coef_spearman
        
        for n in range (len(self.x_df.columns)):
            if fs[n] < coef_spearman:
                bad_features_spearman.append(self.x_df.columns[n])
        
        return bad_features_spearman
    
    
    def get_bad_features_info(self, coef_info):
        
        fs = mutual_info_regression(self.x_df, self.y_values)
        bad_features_info = []
        
        # mean = statistics.mean(fs)
        # threshold = mean*coef_info
        
        for n in range (len(self.x_df.columns)):
            if fs[n] < coef_info:
                bad_features_info.append(self.x_df.columns[n])
                
        return bad_features_info

    
    def get_worst_features(self):
        
        nb_features_to_remove =  len(self.x_df.axes[1]) - len(self.x_df)
        
        if nb_features_to_remove < 0:
            return []
        
        else:
            
            worst_features = []
            columns = []
            fs = []
            
            for column in self.x_df.columns:
                columns.append(column)
            
            for column in self.x_df.columns:
                corr, _ = pearsonr(self.x_df[column], self.y_values)
                fs.append(abs(corr)) 
    
            while len(worst_features) < nb_features_to_remove+1:
                worst_index = np.argmin(fs)
                worst_column = columns[worst_index]    
                fs = np.delete(fs, worst_index)
                columns.remove(worst_column)
                worst_features.append(worst_column)
        
        return worst_features


    def get_bad_features_pvalue(self, pvalue_max):
        
        max_pvalue = 10
        x_df_copy = self.x_df.copy()
        bad_features_pvalue = []

        while max_pvalue > pvalue_max:
    
            new_x = x_df_copy.values
            X2 = sm.add_constant(new_x)
            est = sm.OLS(self.y_values, X2)
            est2 = est.fit()
            p_values = est2.pvalues[1:]
        
            max_pvalue = max(p_values)
    
            max_index = np.argmax(p_values)
            max_variable = x_df_copy.columns[max_index] 
        
            bad_features_pvalue.append(max_variable)
            
            print(p_values)
            print(x_df_copy.columns)
            print(max_pvalue)
            print(max_variable)
            
            
            x_df_copy = x_df_copy.drop([max_variable], axis = 1)
        
        print('pvalue')
        print(bad_features_pvalue)
        return bad_features_pvalue

    
    def get_bad_features_elastic(self):
        
        x_train, x_test, y_train, y_test = self.split_data(0.3, 0)

        ml = ElasticNet(alpha= 10, tol = 0.1, max_iter = 1000)
        ml.fit(x_train, y_train)

        columns = []
        bad_features_elastic = []

        for column in self.x_df.columns:
            columns.append(column)

        n = len(columns)

        for i in range (n):
            if ml.coef_[i] == 0:
                bad_features_elastic.append(columns[i])

        return bad_features_elastic
    
    
    def get_bad_features_r2adj(self):
        
        bad_features_r2adj = []
        drapeau = True
        
        x_df_copy = self.x_df.copy()

        if len(x_df_copy) > len(x_df_copy.columns):
            
            while drapeau :
            
                nb_columns = len(x_df_copy.columns)

                new_x = x_df_copy.values.astype(np.float64)
                X2 = sm.add_constant(new_x).astype(np.float64)
                est = sm.OLS(self.y_values, X2)
                est2 = est.fit()
                r2_adj = est2.rsquared_adj
                worst_r2_adj = r2_adj
                
                for n in range (0, nb_columns):
                    
                    current_column = x_df_copy.columns[n]
                    new_x_df = x_df_copy.drop(current_column, axis = 1)
                    
                    new_x = new_x_df.values
                    X2 = sm.add_constant(new_x)
                    est = sm.OLS(self.y_values.astype(np.float64), X2)
                    est2 = est.fit()
    
                    new_r2_adj = est2.rsquared_adj
                    
                    if new_r2_adj < worst_r2_adj:
                        worst_r2_adj = new_r2_adj
                        worst_column = x_df_copy.columns[n]
                    
                if worst_r2_adj >= r2_adj:
                    drapeau = False 
                    
                else:
                    x_df_copy = x_df_copy.drop(worst_column, axis = 1)
                    bad_features_r2adj.append(worst_column)
                    
        print(bad_features_r2adj)
        return bad_features_r2adj
    