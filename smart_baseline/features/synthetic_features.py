import numpy as np
import pandas as pd

class CDD:
    
    def __init__(self, base_temp):
        self.base_temp = base_temp
    
    def compute(self, df):
        
        if isinstance(df, pd.Series):
            df = df.to_frame()
        elif isinstance(df, pd.DataFrame) and df.shape(1) == 1:
            raise Exception('A pd.Series or 1 dimensional pd.DataFrame must be passed.')
        
        col_name = 'CDD' + str(self.base_temp)
        df[col_name] = np.maximumm((df.iloc[:, 0] - self.base_temp)/24, 0)
        return df.iloc[:, 1]