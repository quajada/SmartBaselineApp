# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:20:58 2022

@author: Bruno Tabet
"""

import numpy as np
import pandas as pd
import psychrolib


class Psychro:
    
    psychrolib.SetUnitSystem(psychrolib.SI)
    cols= ['humratio', 'wbtemp', 'dptemp', 'ppwvap', 'enthma', 'spvolma', 'degsat']

    def __init__(self, temp, rh, p):
        '''

        Parameters
        ----------
        temp : pd.Series
            temperature in C.
        rh : pd.Series
            rh in %.
        p : pd.Series
            pressure in hPa.

        Returns
        -------
        None.

        '''
        
        self.index = temp.index
        self.temp = temp.to_numpy()
        self.rh = rh.divide(100).to_numpy()
        self.rh[self.rh > 1] = 1
        self.rh[self.rh < 0 ] = 0
        self.p = p.multiply(100).to_numpy()
        
    def get_data(self):
        psych_arr = psychrolib.CalcPsychrometricsFromRelHum(self.temp, 
                                                            self.rh,
                                                            self.p)
        psych_arr = np.array(psych_arr)

        return pd.DataFrame(data = psych_arr.T, 
                            index = self.index, 
                            columns = self.cols)
    