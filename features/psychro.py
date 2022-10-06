# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:20:58 2022

@author: Bruno Tabet
"""

import numpy as np
import pandas as pd
import psychrolib
import matplotlib.pyplot as plt

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
        
        # self.p[self.p < 98000] = np.nan
        # self.p = self.p.interpolate()
        # self.p = self.p.to_numpy()
        
        
    def get_data(self):
        
        # test = psychrolib.CalcPsychrometricsFromRelHum(self.temp[0], 
        #                                                self.rh[0],
        #                                                self.p[0])
        
        # nb_psychro_variables = len(test)
        
        nb_psychro_variables = 7
        nb_points = len(self.temp)
        
        psych_arr = tuple(np.zeros(nb_points) for _ in range (nb_psychro_variables))
                
        for i in range(nb_points):

            try:
                new_psych_arr = psychrolib.CalcPsychrometricsFromRelHum(self.temp[i], 
                                                                    self.rh[i],
                                                                    self.p[i])
            except:
                new_psych_arr = np.empty(nb_psychro_variables)
                new_psych_arr[:] = np.NaN
                
            for k in range (nb_psychro_variables):
                psych_arr[k][i] = new_psych_arr[k]
        
        psych_arr = np.array(psych_arr)
        
        return pd.DataFrame(data = psych_arr.T, 
                            index = self.index, 
                            columns = self.cols)