import numpy as np
import pandas as pd

class Aggregator:
    
    weather_agg = {'temp': 'mean',
                   'dwpt': 'mean',
                   'rhum': 'mean',
                   'prcp': 'sum',
                   'snow': 'sum',
                   'wdir': 'mean',
                   'wspd': 'mean',
                   'wpgt': 'mean',
                   'pres': 'mean',
                   'tsun': 'mean',
                   'coco': pd.Series.mode,
                   'humratio': 'mean',
                   'wbtemp': 'mean',
                   'dptemp': 'mean',
                   'ppwvap': 'mean',
                   'enthma': 'mean',
                   'spvolma': 'mean',
                   'degsat': 'mean',
                   'CDD18': 'sum',
                   'CDD19': 'sum',
                   'CDD20': 'sum',
                   'CDD21': 'sum',
                   'CDD22': 'sum',
                   'CDD23': 'sum',
                   'CDD24': 'sum'
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
        
        df_groupped = df.groupby('Group').agg(how)
        
        return df_groupped