# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:14:17 2022

@author: Bruno Tabet
"""

from cleandata import CleanData


def cleandata(x_df, y_df):
    
    c = CleanData(x_df, y_df)
    
    c.remove_bad_columns()
    c.remove_outliers()
    
    return c.x_df, c.y_df




if __name__ == "__main__":
    x_df = pd.read_csv(r'C:\Users\Bruno Tabet\Documents\ENOVA\f.csv')
    y_df = pd.read_csv(r'C:\Users\Bruno Tabet\Documents\ENOVA\b.csv')
    x_df, y_df = cleandata(x_df, y_df)
    
    print(y_df[0])