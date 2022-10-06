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
                    "Scope": sheet["D3"].value,
					"Type": sheet["D4"].value,
                    "Subtype": sheet["D5"].value,
                    "Construction year": sheet["D6"].value,
					"Built-up area": {
						"value": sheet["D7"].value,
						"unit": sheet["E7"].value
						},					
					"Gross floor conditioned area": {
						"value": sheet["D8"].value,
						"unit": sheet["E8"].value
						},		
                    "Number of buildings": sheet["D9"].value,
					"Number of floors including ground": sheet["D10"].value,
					"Number of parking floors/basements": sheet["D11"].value,
					"Cooling source": sheet["D12"].value,
					"Renewable energy sources": sheet["D13"].value,
					"Sewage Treatment Plant (STP)": sheet["D14"].value,
					
					"Address": sheet["I2"].value,
					"City": sheet["I3"].value,
					"Latitude": sheet["I4"].value,
					"Longitude": sheet["I5"].value,
					"Altitude": sheet["I6"].value,
                    
                    "Baseline from": sheet['D16'].value,
                    "Baseline to": sheet['D17'].value,
                    "M&V option": sheet['D18'].value,
                    "Utility": {
                        "name": sheet['D19'].value,
                        "unit": sheet['E19'].value
                        },
                    "M&V scope":sheet["D20"].value,
                    "Savings": { 
                        "value": sheet["D21"].value,
                        "unit": sheet["E21"].value,
                        },
                    "Currency": sheet["D22"].value,
                    "Base tariff without VAT": {
                        "value": sheet["D23"].value,
                        "unit": sheet["E23"].value
                        }
					}
    
    
    def table_to_df(self, ws, table: str):
        
        # Get table range
        if table in ws.tables:
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
            if len(df.columns)>2:
                if not df.iloc[-1, 2]:
                    df = df.drop(df.index[-1])
        
        else:
            df = pd.DataFrame()
        
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
        df_weather = df_weather.drop_duplicates()
        
        one_hour = df_weather.index[1] - df_weather.index[0]
        df2 = {}
        new_row = []
        for column in df_weather.columns:
            df2[column] = None
            new_row.append(None)
        
        
        for i in range (0, len(df_weather)-1):
            if df_weather.index[i+1] - df_weather.index[i] != one_hour:
                index = df_weather.index[i]
                nb_of_hours = 2
                for j in range(1, nb_of_hours):
                    new_row[-2] = df_weather.loc[index]['From (incl)'] + one_hour*j
                    new_row[-1] = df_weather.loc[index]['To (excl)'] + one_hour*j
                    df_weather.loc[index + one_hour] = new_row
                    # print(df_weather.loc[index + one_hour])
                    # print(index + one_hour)
        
        df_weather = df_weather.sort_index()
        
        df_weather.replace({pd.NaT: None}, inplace=True)
        
        for i in range (0, len(df_weather.columns)-2):
            df_weather[df_weather.columns[i]] = df_weather[df_weather.columns[i]].interpolate()
 
        
        # Psychrometric data
        df_psych = Psychro(df_weather['temp'],
                           df_weather['rhum'],
                           df_weather['pres'])

        df_psych = df_psych.get_data()

        # Merge weather features
        weather_features = df_weather.merge(df_psych, left_index=True, right_index=True)
        # weather_features[weather_features.select_dtypes('float64').columns] = weather_features.select_dtypes('float64').astype('float16')

        
        base_temps = [18, 19, 20, 21, 22, 23, 24]
        
        # print(weather_features)
        
        for bt in base_temps:
            cdd = CDD(bt) 
            cdd_df = cdd.compute(weather_features['temp'])
            hdd = HDD(bt)
            hdd_df = hdd.compute(weather_features['temp'])
            # weather_features[weather_features.select_dtypes('float64').columns] = weather_features.select_dtypes('float64').astype('float16')
            weather_features = weather_features.merge(cdd_df, left_index=True, right_index=True)
            # weather_features[weather_features.select_dtypes('float64').columns] = weather_features.select_dtypes('float64').astype('float16')   
            weather_features = weather_features.merge(hdd_df, left_index=True, right_index=True)
            # weather_features[weather_features.select_dtypes('float64').columns] = weather_features.select_dtypes('float64').astype('float16')  
            
        
        if len(self.features.columns) > 2:
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
        features = weather_features_resampled
        if len(self.features.columns) > 2:
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
        
        y_df = y_df.drop(['Timedelta'], axis = 1)
        
        timedelta = np.zeros(len(y_df))
        for i in range(len(timedelta)):
            timedelta[i] = (y_df['To (excl)'][i]-y_df['From (incl)'][i]).total_seconds()/3600
        y_df['Timedelta'] = timedelta

        baseline = self.get_baseline(normalize = False)

        return x_df, y_df, baseline
    
    
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