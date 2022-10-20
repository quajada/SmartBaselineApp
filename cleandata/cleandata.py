# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:14:10 2022

@author: Bruno Tabet
"""

import numpy as np
import pandas as pd

def _string(h):
    if type(h) == str:
        return str(h)
    else:
        return 'most'


class Aggregator:
    
    weather_agg = {'temp': ['mean', 'max', 'min','std'],
                   'dwpt': ['mean', 'max', 'min','std'],
                   'rhum': ['mean', 'max', 'min','std'],
                   'prcp': ['sum','mean', 'max', 'min','std'],
                   'snow': ['sum', 'mean', 'max', 'min','std'],
                   'wdir': ['mean', 'max', 'min','std'],
                   'wspd': ['mean', 'max', 'min','std'],
                   'wpgt': ['mean', 'max', 'min','std'],
                   'pres': ['mean', 'max', 'min','std'],
                   'tsun': ['mean', 'max', 'min','std'],
                   'coco': [pd.Series.mode],
                   'humratio': ['mean', 'max', 'min','std'],
                   'wbtemp': ['mean', 'max', 'min','std'],
                   'dptemp': ['mean', 'max', 'min','std'],
                   'ppwvap': ['mean', 'max', 'min','std'],
                   'enthma': ['mean', 'max', 'min','std'],
                   'spvolma': ['mean', 'max', 'min','std'],
                   'degsat': ['mean', 'max', 'min','std'],
                   'CDD18': ['sum'],
                   'CDD19': ['sum'],
                   'CDD20': ['sum'],
                   'CDD21': ['sum'],
                   'CDD22': ['sum'],
                   'CDD23': ['sum'],
                   'CDD24': ['sum'],
                   'HDD18': ['sum'],
                   'HDD19': ['sum'],
                   'HDD20': ['sum'],
                   'HDD21': ['sum'],
                   'HDD22': ['sum'],
                   'HDD23': ['sum'],
                   'HDD24': ['sum'],

                   }
    


    def group_between_dates(df_base: pd.DataFrame(), df: pd.DataFrame(), how: dict):
        
        '''
        df_base: pd.DataFrame to use for grouping
        df: pd.DataFrame to group
        how: dict {column: method} to pass to .agg()
        '''
        
        df['Group'] = np.nan
        
        for i in range(len(df_base)):
            
            start = df_base.iloc[i, 0]
            end = df_base.iloc[i, 1]
            group = int(df_base.index[i])
    
            mask = (df['From (incl)'] >= start) & (df['To (excl)'] <= end)
    
            df.loc[mask, 'Group'] = group
        
        
        how_to_agg = {f: [(_string(h) + '_' + str(f), h) for h in h_list] for f, h_list in how.items()}
    
        df_groupped = df.groupby('Group').agg(how_to_agg)
        df_groupped.columns = df_groupped.columns.droplevel()     
        
        return df_groupped





class CleanDataFrame:
    
    def __init__(self, x_df):
        self.x_df = x_df


class CleanColumns(CleanDataFrame):

    def __init__(self, x_df):
        super().__init__(x_df)
        
    def remove_bad_columns(self):
        cols_to_drop = []
        for col in self.x_df.columns:
            if len(self.x_df[col].unique()) <= len(self.x_df)-2:
                cols_to_drop.append(col)
        self.x_df = self.x_df.drop(cols_to_drop, axis=1)
        return self.x_df

    def remove_nan(self):
        self.x_df = self.x_df.dropna(axis = 1, how = 'all')
        return self.x_df
        
    def fill_nan(self):
        self.x_df.interpolate()
        return self.x_df

    def remove_duplicates(self):
        self.x_df = self.x_df.T.drop_duplicates().T
        return self.x_df
        

class CleanRows(CleanDataFrame):
    
    def __init__(self, x_df, y_df):
        super().__init__(x_df)
        self.y_df = y_df
        
        
    def get_outliers_quantile(self, quantile):
        q99 = self.y_df.quantile(1-quantile)
        q01 = self.y_df.quantile(quantile)
        cache = (self.y_df <= q99) & (self.y_df >= q01)
        bad_rows = [i for i in cache.index if cache[i] == False]
        return bad_rows


    def get_outliers_zvalue(self, n_std):
        mean = self.y_df.mean()
        sd = self.y_df.std()
        cache = (self.y_df <= mean+(n_std*sd))     
        bad_rows = [i for i in cache.index if cache[i] == False]
        return bad_rows
    
    
    def remove_rows(self, bad_rows):
        self.x_df = self.x_df.drop(bad_rows)
        self.y_df = self.y_df.drop(bad_rows)
        return self.x_df, self.y_df