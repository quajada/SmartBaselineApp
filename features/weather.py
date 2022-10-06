# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 16:19:40 2022

@author: Bruno Tabet
"""

import pandas as pd
from meteostat import Stations, Hourly
import datetime

class WeatherStation:
    
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon
        
    def get_station(self, n = 1):
        # USING STATION CLASS

        # Get nearby weather stations
        stations = Stations()
        stations = stations.nearby(self.lat, self.lon)
        self.station = stations.fetch(n)
        return self.station
    
    def get_data(self, start, end) -> pd.DataFrame:
        
        print(start, end)
        # Setup api
        # end = datetime.datetime.now()
        end += datetime.timedelta(hours = 1)
        start += datetime.timedelta(hours =-1) 
        print(end)
        weather_api = Hourly(self.station, start, end)
        
        # Get data for period
        weather_df = weather_api.fetch()
        
        print(weather_df)
        # Standardize
        weather_df['From (incl)'] = weather_df.index
        weather_df['To (excl)'] = weather_df['From (incl)'] + pd.DateOffset(hours=1)

        return weather_df