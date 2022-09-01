# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 11:21:15 2022

@author: Bruno Tabet
"""

import pandas as pd
import numpy as np

class CDD:
    
    def __init__(self, base_temp):
        self.base_temp = base_temp
    
    def compute(self, df):
        
        if isinstance(df, pd.Series):
            df = df.to_frame()
        elif isinstance(df, pd.DataFrame) and df.shape(1) == 1:
            raise Exception('A pd.Series or 1 dimensional pd.DataFrame must be passed.')
        
        col_name = 'CDD' + str(self.base_temp)
        df[col_name] = np.maximum((df.iloc[:, 0] - self.base_temp)/24, 0)
        return df.iloc[:, 1]


class HDD:
    
    def __init__(self, base_temp):
        self.base_temp = base_temp
    
    def compute(self, df):
        
        if isinstance(df, pd.Series):
            df = df.to_frame()
        elif isinstance(df, pd.DataFrame) and df.shape(1) == 1:
            raise Exception('A pd.Series or 1 dimensional pd.DataFrame must be passed.')
        
        col_name = 'HDD' + str(self.base_temp)
        df[col_name] = np.maximum((self.base_temp - df.iloc[:, 0])/24, 0)
        return df.iloc[:, 1]


class SyntheticFeatures:
    
    def __init__(self, x_df):
        self.x_df = x_df
    
    def create_sqrt_and_squared(self):
        for column in self.x_df.columns:
            if column != "coco" and column != "Group" and 'times' not in column and 'max_' not in column and 'min_' not in column and 'std_' not in column:
                self.x_df['sqrt_'+column] = self.x_df[column]**0.5
                self.x_df[column+'_squared'] = self.x_df[column]**2
        return self.x_df
                
    def create_inverse(self, columns):
        if type(columns) == list:
            for column in columns:
                self.x_df['inv_'+column] = self.x_df[column]**(-1)
        
        if type(columns) == str:
            self.x_df['inv_'+columns] = self.x_df[columns]**(-1)
        return self.x_df
    
    
    def create_sqrt(self, columns):
        if type(columns) == list:
            for column in columns:
                self.x_df['sqrt_'+column] = self.x_df[column]**(1/2)
        
        if type(columns) == str:
            self.x_df['sqrt_'+columns] = self.x_df[columns]**(1/2)
        return self.x_df
    
    
    def create_squared(self, columns):
        if type(columns) == list:
            for column in columns:
                self.x_df['squared_'+column] = self.x_df[column]**(2)
        
        if type(columns) == str:
            self.x_df['squared_'+columns] = self.x_df[columns]**(2)
        return self.x_df
    

    def create_products(self):
        columns = self.x_df.columns
        n = len(columns)
        for i in range(n):
            columni = columns[i]
            if 'CDD' not in columni and 'coco' not in columni and 'sqrt' not in columni and 'squared' not in columni:
                for j in range(i+1, n):
                    columnj = columns[j]
                    if 'CDD' not in columnj and 'coco' not in columnj and 'sqrt' not in columnj and 'squared' not in columnj:
                        self.x_df[columni+'_times_'+columnj]= self.x_df[columni]*self.x_df[columnj]
        return self.x_df
    
    
    def remove_str_beginning(self, string):
        bad_columns = []
        n = len(string)
        for column in self.x_df.columns:
            if column[0:n] == string:
                bad_columns.append(column)
        print(bad_columns)
        self.x_df = self.x_df.drop([feature for feature in bad_columns], axis = 1)