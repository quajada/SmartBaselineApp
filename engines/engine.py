# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 12:57:24 2022

@author: Bruno Tabet
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import stats
from statistics import mean, stdev
from itertools import combinations
import random

class Engine:
    
    def __init__(self, x_df, y_df, combinations, max_variables, nb_folds, test_size):
        self.x_df = x_df
        self.y_df = y_df
        self.combinations = combinations
        self.max_variables = max_variables
        self.nb_folds = nb_folds
        self.test_size = test_size
        self.results = {}
        self.IPMVP_results = {}
        self.best_results = {}
    
    
    def compute_pval_and_tval(self, ml, X_test, y_test, y_pred):
        params = np.append(ml.intercept_,ml.coef_)
        new_X = np.append(np.ones((len(X_test),1)), X_test, axis=1)
        M_S_E = (sum((y_test-y_pred)**2))/(len(new_X)-len(new_X[0]))
        v_b = M_S_E*(np.linalg.inv(np.dot(new_X.T,new_X)).diagonal())
        s_b = np.sqrt(v_b)
        t_val = params/ s_b
        p_val =[2*(1-stats.t.cdf(np.abs(i),(len(new_X)-len(new_X[0])))) for i in t_val]
        p_val = np.round(p_val,3)
        p_val
        return p_val.tolist(), t_val.tolist()
    
    def compute_cv_rmse(self, std_dev, X_test, y_test, y_pred):
        cv_rmse = std_dev/mean(y_test)
        return cv_rmse
    

    def _compute_y_pred(self, coefs, combination):
        y_pred = np.ones(len(self.y_df['Normalized baseline']))*coefs[0]
        for i in range (len(combination)):
            y_pred += self.x_df[combination[i]].values*coefs[i+1]
        # timedelta = self.y_df['Timedelta'].tolist()
        # for k in range (len(y_pred)):
        #     y_pred[k] = y_pred[k]*timedelta[k]
        return y_pred


    def _compute_std_dev(self, y_test, y_pred, nb_of_var):
        y = y_test - y_pred
        # timedelta = self.y_df['Timedelta'].tolist()
        # for k in range(len(y)):
        #     y[k] = y[k] * timedelta[k]
        y = y**2
        std_dev = np.sqrt( sum(y) / (len(y)-nb_of_var-1) )
        return std_dev
        
            
    


    def compute_cross_validation2(self):
        
        nb_iterations = self.nb_folds
            
        nombre_total = len(self.combinations)
        itera = 0
        
        for combination in self.combinations:
            
            itera += 1
            
            if itera == int(nombre_total/4):
                print('UN QUART')
            
            self.results[combination] = {}
            self.results[combination]['r2_cv_test'] = []
            self.results[combination]['pval_cv'] = {}
            self.results[combination]['tval_cv'] = {}
            for i in range (1+len(combination)):
                self.results[combination]['pval_cv'][i] = []
                self.results[combination]['tval_cv'][i] = []
            self.results[combination]['std_dev_cv_test'] = []
            self.results[combination]['cv_rmse_cv'] = []

            from itertools import combinations
            
            index_combi = list(combinations(self.x_df.index.tolist(), self.test_size))
            for n in range(nb_iterations):

                new_indexes = random.choice(index_combi)
                list_indexes = [index for index in new_indexes]
                # not_list_indexes = self.x_df.index.tolist()
                # for index in list_indexes:
                #     not_list_indexes.remove(index)
                    
                # new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(self.x_df, self.y_df['Normalized baseline'], test_size= self.test_size/len(self.x_df), shuffle=True) 
                new_X_test = self.x_df.loc[list_indexes]
                new_X_train = self.x_df.drop(list_indexes)
                
                new_Y_df_test = self.y_df.loc[list_indexes]
                new_Y_df_train = self.y_df.drop(list_indexes)
                
                new_y_test = new_Y_df_test['Normalized baseline'].astype(np.float64).tolist()
                new_y_train = new_Y_df_train['Normalized baseline'].astype(np.float64).tolist()
                
                new_X_train = new_X_train[[combination[i] for i in range(len(combination))]]
                new_X_train = new_X_train.values.astype(np.float64)
                
                new_X_test = new_X_test[[combination[i] for i in range(len(combination))]]
                new_X_test = new_X_test.values.astype(np.float64)
                
                ml = LinearRegression()
                ml.fit(new_X_train, new_y_train)
                y_pred = ml.predict(new_X_test)
                # timedelta = new_Y_df_test['Timedelta'].tolist()
                # for k in range (len(y_pred)):
                #     y_pred[k] = y_pred[k] * timedelta[k]

                # pvalues, tvalues = self.compute_pval_and_tval(ml, new_X_test, new_y_test, y_pred)
                
                std_dev = self._compute_std_dev(new_y_test, y_pred, len(combination))
                
                # print(r2_score(new_y_test, y_pred))
                
                self.results[combination]['r2_cv_test'].append(r2_score(new_y_test, y_pred))
                self.results[combination]['std_dev_cv_test'].append(std_dev)
                
                # for i in range (len(pvalues)):
                #     self.results[combination]['pval_cv'][i].append(pvalues[i])
                                            
                # for i in range (len(tvalues)):
                #     self.results[combination]['tval_cv'][i].append(abs(tvalues[i]))
                # self.results[combination]['cv_rmse_cv'].append(self.compute_cv_rmse(std_dev, new_X_test, new_y_test, y_pred))

            self.results[combination]['r2_cv_test'] = mean(self.results[combination]['r2_cv_test']) - 0*stdev(self.results[combination]['r2_cv_test'])
            self.results[combination]['std_dev_cv_test'] = mean(self.results[combination]['std_dev_cv_test']) + 0*stdev(self.results[combination]['std_dev_cv_test'])
            # self.results[combination]['cv_rmse_cv'] = mean(self.results[combination]['cv_rmse_cv']) + 0*stdev(self.results[combination]['cv_rmse_cv'])

            # pval = []        
            # tval = []
            # for i in range (len(combination)+1):
            #     pval.append(mean(self.results[combination]['pval_cv'][i])+0*stdev(self.results[combination]['pval_cv'][i]))
            #     tval.append(median(self.results[combination]['tval_cv'][i])-0*stdev(self.results[combination]['tval_cv'][i]))
            
            # self.results[combination]['pval_cv'] = pval
            # self.results[combination]['tval_cv'] = tval

        return self.results
    
    
    





    
    def compute_cross_validation(self):
        
        nb_iterations = self.nb_folds
            
        nombre_total = len(self.combinations)
        itera = 0
        
        for combination in self.combinations:
            
            itera += 1
            
            if itera == int(nombre_total/4):
                print('UN QUART')
            
            self.results[combination] = {}
            self.results[combination]['r2_cv_test'] = []
            self.results[combination]['pval_cv'] = {}
            self.results[combination]['tval_cv'] = {}
            for i in range (1+len(combination)):
                self.results[combination]['pval_cv'][i] = []
                self.results[combination]['tval_cv'][i] = []
            self.results[combination]['std_dev_cv_test'] = []
            self.results[combination]['cv_rmse_cv'] = []

            for n in range(nb_iterations):
        
                new_X_train, new_X_test, new_y_df_train, new_y_df_test = train_test_split(self.x_df, self.y_df, test_size= self.test_size, shuffle=True) 
        
                new_X_train = new_X_train[[combination[i] for i in range(len(combination))]]
                new_X_train = new_X_train.values.astype(np.float64)
                
                new_X_test = new_X_test[[combination[i] for i in range(len(combination))]]
                new_X_test = new_X_test.values.astype(np.float64)
                
                # new_X_train, new_X_test, new_y_train, new_y_test = train_test_split(self.x_df, self.y_df['Normalized baseline'], test_size= self.test_size/len(self.x_df), shuffle=True) 
                
                new_y_test = new_y_df_test['Normalized baseline'].astype(np.float64).tolist()
                new_y_train = new_y_df_train['Normalized baseline'].astype(np.float64).tolist()
                
                
                ml = LinearRegression()
                ml.fit(new_X_train, new_y_train)
                y_pred = ml.predict(new_X_test)
                # timedelta = new_Y_df_test['Timedelta'].tolist()
                # for k in range (len(y_pred)):
                #     y_pred[k] = y_pred[k] * timedelta[k]

                # pvalues, tvalues = self.compute_pval_and_tval(ml, new_X_test, new_y_test, y_pred)
                
                
                # pvalues, tvalues = self.compute_pval_and_tval(ml, new_X_test, new_y_test, y_pred)
                std_dev = self._compute_std_dev(new_y_test, y_pred, len(combination))
                
                # print(r2_score(new_y_test, y_pred))
                
                self.results[combination]['r2_cv_test'].append(r2_score(new_y_test, y_pred))
                self.results[combination]['std_dev_cv_test'].append(std_dev)

                # for i in range (len(pvalues)):
                #     self.results[combination]['pval_cv'][i].append(pvalues[i])
                                            
                # for i in range (len(tvalues)):
                #     self.results[combination]['tval_cv'][i].append(abs(tvalues[i]))
                # self.results[combination]['cv_rmse_cv'].append(self.compute_cv_rmse(std_dev, new_X_test, new_y_test, y_pred))

            self.results[combination]['r2_cv_test'] = mean(self.results[combination]['r2_cv_test']) - 0*stdev(self.results[combination]['r2_cv_test'])
            self.results[combination]['std_dev_cv_test'] = mean(self.results[combination]['std_dev_cv_test']) + 0*stdev(self.results[combination]['std_dev_cv_test'])
            # self.results[combination]['cv_rmse_cv'] = mean(self.results[combination]['cv_rmse_cv']) + 0*stdev(self.results[combination]['cv_rmse_cv'])

            # pval = []        
            # tval = []
            # for i in range (len(combination)+1):
            #     pval.append(mean(self.results[combination]['pval_cv'][i])+0*stdev(self.results[combination]['pval_cv'][i]))
            #     tval.append(median(self.results[combination]['tval_cv'][i])-0*stdev(self.results[combination]['tval_cv'][i]))
            
            # self.results[combination]['pval_cv'] = pval
            # self.results[combination]['tval_cv'] = tval

        return self.results

    
    
    
    
    def are_combinations_IPMVP_consistently_compliant(self):
    
        for combination in self.results:
            isCompliant = True
            
            if self.results[combination]['r2_cv_test'] < 0.75:
                isCompliant = False

            if self.results[combination]['cv_rmse_cv'] >= 0.2:
                isCompliant = False
            
            pvalues = self.results[combination]['pval_cv']
            for pvalue in pvalues:
                if  pvalue >= 0.1:
                    isCompliant = False

            tvalues = self.results[combination]['tval_cv']
            for tvalue in tvalues:
                if abs(tvalue) <= 2:
                    isCompliant = False

            
            self.results[combination]['IPMVP_consistently_compliant'] = isCompliant
        
        return self.results
    
    




    def are_combinations_IPMVP_compliant(self):
        
        for combination in self.results:
            
            new_x_df = self.x_df[[combination[i] for i in range(len(combination))]]
            new_x = new_x_df.values.astype(np.float64)
        
            X2 = sm.add_constant(new_x).astype(np.float64)
            est = sm.OLS(self.y_df['Normalized baseline'].astype(np.float64).tolist(), X2)
            est2 = est.fit()
            coefs = est2.params.tolist()
            
            self.results[combination]['r2'] = est2.rsquared
            self.results[combination]['pval'] = est2.pvalues.tolist()
            self.results[combination]['tval'] = est2.tvalues.tolist()
            self.results[combination]['coefs'] = est2.params.tolist()
            self.results[combination]['AIC'] = est2.aic
            self.results[combination]['BIC'] = est2.bic
            self.results[combination]['cv_rmse'] = np.sqrt(est2.mse_resid)/mean(self.y_df['Normalized baseline'])
            self.results[combination]['AIC_adj'] = -2*est2.llf + 2*(self._score_combination(combination)+1)
            self.results[combination]['intercept'] = coefs[0]
            self.results[combination]['slopes'] = coefs[1:]
            self.results[combination]['size'] = len(combination)
            
            y_pred = self._compute_y_pred(coefs, combination)
            
            self.results[combination]['y_pred'] = y_pred
            self.results[combination]['std_dev'] = self._compute_std_dev(self.y_df['Normalized baseline'], y_pred, len(combination))
            
            isCompliant = True
            if est2.rsquared < 0.75:
                isCompliant = False
            
            pvalues = est2.pvalues.tolist()
            for pvalue in pvalues:
                if  pvalue >= 0.1:
                   isCompliant = False

            tvalues = est2.tvalues.tolist()
            for tvalue in tvalues:
                if abs(tvalue) <= 2:
                    isCompliant = False
                    
            if np.sqrt(est2.mse_resid)/mean(self.y_df['Normalized baseline']) > 0.2:
                isCompliant = False

            self.results[combination]['IPMVP_compliant'] = isCompliant
    
        return self.results

        



    def get_df_results(self):
        columns = ['combinations', 'r2', 'std_dev', 'r2_cv_test', 'std_dev_cv_test', 'intercept', 'pval', 'tval', 'cv_rmse', 'IPMVP_compliant', 'AIC', 'AIC_adj', 'size']
        self.results_df = pd.DataFrame(index = [i for i in range (len(self.results))], columns = columns)
        
        i = 0
        columns_list = ['pval', 'tval']
        
        for combination in self.results:
            self.results_df.iloc[i]['combinations'] = combination
            for column in columns[1:]:
                if column in columns_list:
                    round_values = self.results[combination][column].copy()
                    for k in range (len(round_values)):
                        round_values[k] = round(round_values[k], 4)
                    self.results_df.iloc[i][column] = round_values
                else:
                    self.results_df.iloc[i][column] = self.results[combination][column]
            i+=1
        
        columns_float = ['r2', 'std_dev', 'r2_cv_test', 'std_dev_cv_test', 'intercept', 'cv_rmse', 'AIC', 'AIC_adj', 'size']
        for column in columns_float:
            self.results_df[column] = pd.to_numeric(self.results_df[column])
            self.results_df[column] = self.results_df[column].round(3)
        
        return self.results_df



    def show_top_results(self, criteria, nb_variables, nb_top):
        
        best_combinations = self.get_top_results(criteria, nb_variables, nb_top)
        
        self.best_results = {}
        
        for combination in best_combinations:
            
            self.best_results[combination] = {}
            
            new_x_df = self.x_df[[combination[i] for i in range(len(combination))]]
            new_x = new_x_df.values.astype(np.float64)
            
            if combination != ():

                X2 = sm.add_constant(new_x).astype(np.float64)
                est = sm.OLS(self.y_df['Normalized baseline'].astype(np.float64).tolist(), X2)
                est2 = est.fit()
                
                self.best_results[combination]['r2'] = est2.rsquared
                self.best_results[combination]['pval'] = est2.pvalues.tolist()
                self.best_results[combination]['tval'] = est2.tvalues.tolist()
                self.best_results[combination]['coefs'] = est2.params.tolist()
                self.best_results[combination]['AIC'] = est2.aic
                self.best_results[combination]['BIC'] = est2.bic
                self.best_results[combination]['std_dev'] = np.sqrt(est2.mse_resid)
                self.best_results[combination]['cv_rmse'] = np.sqrt(est2.mse_resid)/mean(self.y_df['Normalized baseline'])
                self.best_results[combination]['AIC_adj'] = -2*est2.llf + 2*(self._score_combination(combination)+1)
        
        print('')
        print('THESE ARE THE TOP BEST RESULTS')
        for combination in self.best_results:
            print(combination)
            
        for combination in self.best_results:
            print('')
            print(combination)
            for key in self.best_results[combination] :
                print(key, ' : ', self.best_results[combination][key])
        
        return self.best_results

    
    def _score_combination(self, combination):
        score = 0
        
        psycho_variables = ['humratio', 'wbtemp', 'dptemp', 'ppwvap', 'enthma', 'spvolma', 'degsat']
        
        for variable in combination:
            
            if variable in psycho_variables:
                score += 0.25
                
            elif '_' not in variable:
                score += 0
            
            else:
                score += 1
                
        return score