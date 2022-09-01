# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:24:21 2022

@author: Bruno Tabet
"""

import openpyxl
import pandas as pd
import numpy as np

from features.weather import WeatherStation
from features.psychro import Psychro
from features.synthetic_features import CDD, HDD
from cleandata.cleandata import Aggregator, CleanColumns

class ReadExcel:
    
    def __init__(self, file_path):
        
        # Open Workbook
        self.wb = openpyxl.load_workbook(filename = file_path, data_only = True)
        sheet = self.wb["Project"]
        
        # Extract params from excel file
        self.data = {
					"Project name": sheet["D2"].value,
					"Building type": sheet["D3"].value,
					"Building subtype": sheet["D4"].value,
					"Built-up area": {
						"value": sheet["D5"].value,
						"unit": sheet["E5"].value
						},					
					"Gross floor conditioned area": {
						"value": sheet["D6"].value,
						"unit": sheet["E6"].value
						},		
					"Number of floors including ground": sheet["D7"].value,
					"Number of parking floors/basements": sheet["D8"].value,
					"Cooling source": sheet["D9"].value,
					"Renewable energy sources": sheet["D10"].value,
					"Sewage Treatment Plant (STP)": sheet["D11"].value,
					
					"Locator type": sheet["I2"].value,
					"City": sheet["I3"].value,
					"Latitude": sheet["I4"].value,
					"Longitude": sheet["I5"].value,
					"Altitude": sheet["I6"].value,
					}
    
    
    
    def table_to_df(self, ws, table: str):
        
        # Get table range
        cells = ws[ws.tables[table].ref]
        
        # Iterate through table range cells and create a list of lists
        cells_value = list()
    
        for cell in cells:
            row = list()
            for i in range(0, len(cell)):
                row.append(cell[i].value)
    
            cells_value.append(row)
    
        # Load the list to a dataframe
        df = pd.DataFrame(cells_value)
        # Grab the first row for the header
        df_header = df.iloc[0]
        # Get the data except the 1st row
        df = df[1:]
        # Set the 1st row as header
        df.columns = df_header
        # Reset df index to Timestamp
        df.reset_index(drop=True, inplace=True)
        
        # Reset index to Timestamp
        #df.set_index('Timestamp', inplace=True)
        # Converting index to datetime
        #df.index = pd.to_datetime(df.index)
        
        # Remove last row if None
        if not df.iloc[-1, 2]:
            df = df.drop(df.index[-1])
        
        return df
    
    
    def preprocess_data(self, path):
        
        # Weather data
        w = WeatherStation(self.data['Latitude'], self.data['Longitude'])
        weather_station = w.get_station()
        print(f'Weather station selected: {weather_station.name}')
        
        baseline = self.get_baseline()
        self.start = baseline['From (incl)'][0]
        self.end = baseline['To (excl)'][len(baseline)-1]
        
        df_weather = w.get_data(self.start, self.end)
        
        # Psychrometric data
        df_psych = Psychro(df_weather['temp'],
                           df_weather['rhum'],
                           df_weather['pres'])
        print('AAAAAAAA')
        
        df_psych = df_psych.get_data()
        
        print('BBBBBBBBBB')
        
        
        # Merge weather features
        weather_features = df_weather.merge(df_psych, left_index=True, right_index=True)
        
        # Add CDD
        base_temps = [18, 19, 20, 21, 22, 23, 24]
        for bt in base_temps:
            cdd = CDD(bt)
            cdd_df = cdd.compute(weather_features['temp'])
            hdd = HDD(bt)
            hdd_df = hdd.compute(weather_features['temp'])
            weather_features = weather_features.merge(cdd_df, left_index=True, right_index=True)
            weather_features = weather_features.merge(hdd_df, left_index=True, right_index=True)
        
        for feature in self.features.columns[2:]:
            self.features[feature] = self.features[feature].astype(float)
            
        custom_features_resampled = Aggregator.group_between_dates(self.baseline, 
                                                                self.features,
                                                                {col: ['sum', 'mean', 'max', 'min','std'] for col in self.features.columns[2:]})
        
        float_cols = weather_features.select_dtypes(include=['float64', 'number']).columns
        
        weather_features.loc[:, float_cols] = weather_features.loc[:, float_cols].fillna(0)
        
        weather_features_resampled = Aggregator.group_between_dates(self.baseline, 
                                                                 weather_features, 
                                                                 Aggregator.weather_agg)
        
        # Merge features
        features = custom_features_resampled.merge(weather_features_resampled, left_index=True, right_index=True)
        
        for column in features.columns:
            # print(column)
            bad_indexes = features[column][features[column].apply(lambda x: not isinstance(x, float))].index.values
            df1 = features[column][features[column].apply(lambda x: isinstance(x, float))]
            df1 = df1.to_frame(name = column)
            df2 = pd.DataFrame(0, columns = [column], index = bad_indexes)
            df3 = pd.concat([df1, df2])
            features[column] = df3[column]
            
        features = features.astype(np.float64)

        x_df = features
        y_df = baseline
        
        # clean_col = CleanColumns(x_df)
        # x_df = clean_col.remove_bad_columns()

        return x_df, y_df
    
    
    @property
    def baseline(self):
        if hasattr(self, '_baseline'):
            return self._baseline
        return self.get_baseline()
    
    def get_baseline(self, normalize=True):
        df = self.table_to_df(self.wb['Baseline'], 'Baseline')
        if normalize:
            df['Timedelta'] = df['To (excl)'] - df['From (incl)']
            df['Normalized baseline'] = df.iloc[:, 2] / (df['Timedelta'] / np.timedelta64(1, 'h'))
        self._baseline = df
        return df
    
    @property
    def features(self):
        if hasattr(self, '_features'):
            return self._features
        return self.get_features()
    
    def get_features(self):
        self._features = self.table_to_df(self.wb['Variables'], 'Variables')
        return self._features