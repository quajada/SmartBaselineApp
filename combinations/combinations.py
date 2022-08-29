# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 15:04:52 2022

@author: Bruno Tabet
"""

import itertools
import time

class Combinations:
    

    def __init__(self, list_variables, max_nb_variables):
        self.list_variables = list_variables
        self.max_nb_variables = max_nb_variables
        self.combinations = []
        self.impossible_combs = {}

    def compute_combinations(self, x_df):
        
        corr = x_df.corr()
        corr = corr[(abs(corr) > 0.9) & (abs(corr) != 1)]
        self.impossible_combs = {col: set(corr[col].dropna().index) for col in corr.columns}
        
        return self.create_combinations()
       
    
    def create_combinations(self):
                
        start = time.time()
        final_comb = []
        for c in range(1, 1+self.max_nb_variables):
            combs = list(itertools.combinations(self.list_variables, int(c)))
            final_comb.extend(self._filter_combinations(combs, self.impossible_combs))
            
        print(time.time()-start)
                
        self.combinations = final_comb
        
        return self.combinations
    
    
    def _filter_combinations(self, combs, imposible_combs):

        filtered_comb = []
    
        for comb in combs:
            append = True
            for i in comb:
                if set(comb) & imposible_combs[i]:
                    #print(comb, i, set(comb) & imposible_combs[i])
                    append = False
                    break
            if append:
                filtered_comb.append(comb)
    
        print(f'Reduced from {len(combs)} to {len(filtered_comb)}')
        return filtered_comb
        
    
    
    def create_first_combinations(self):
        for n in range (1, 1 + self.max_nb_variables):
            for combination in itertools.combinations(self.list_variables, n):
                self.combinations.append(combination)
        return self.combinations

    
    def remove_CDD_combinations(self):
        
        bad_combinations = []
        
        for combination in self.combinations:
            
            temp = []
            remove = False
            
            for variable in combination:
                if 'CDD' in variable:
                    i = 0
                    foundCDD = False
                    
                    while not foundCDD :
                        if variable[i:i+3] == 'CDD':
                            number_position = i+3
                            foundCDD = True
                        else:
                            i+=1
                    new_temp = variable[number_position:number_position+2]
                    
                    if len(temp)>0 and new_temp not in temp:         
                        remove = True
                    
                    else:
                        temp.append(new_temp)
                    
            if remove == True:
                bad_combinations.append(combination)
                
        for combination in bad_combinations:
            self.combinations.remove(combination)
        
        return self.combinations


    def remove_HDD_combinations(self):
        
        bad_combinations = []
        
        for combination in self.combinations:
            
            temp = []
            remove = False
            
            for variable in combination:
                if 'HDD' in variable:
                    i = 0
                    foundHDD = False
                    
                    while not foundHDD :
                        if variable[i:i+3] == 'HDD':
                            number_position = i+3
                            foundHDD = True
                        else:
                            i+=1
                    new_temp = variable[number_position:number_position+2]
                    
                    if len(temp)>0 and new_temp not in temp:         
                        remove = True
                    
                    else:
                        temp.append(new_temp)
                    
            if remove == True:
                bad_combinations.append(combination)
                
        for combination in bad_combinations:
            self.combinations.remove(combination)
        
        return self.combinations            