from excel_funcs import ReadExcel
from features.weather import WeatherStation
from features.psychro import Psychro
from features.synthetic_features import CDD
from cleaner.cleaner import Aggregator


def preprocess_data(path):
    
    e = ReadExcel(path)
    
    # Weather data
    w = WeatherStation(e.lat, e.lon)
    weather_station = w.get_station()
    print(f'Weather station selected: {weather_station.name}')
    df_weather = w.get_data(e.start, e.end)
    
    # Psychrometric data
    df_psych = Psychro(df_weather['temp'],
                       df_weather['rhum'],
                       df_weather['pres'])
    df_psych = df_psych.get_data()
    
    # Merge weather features
    weather_features = df_weather.merge(df_psych, left_index=True, right_index=True)
    
    # Add CDD
    base_temps = [18, 19, 20, 21, 22, 23, 24]
    for bt in base_temps:
        cdd = CDD(bt)
        cdd_df = cdd.compute(weather_features['temp'])
        weather_features = weather_features.merge(cdd_df, left_index=True, right_index=True)
        
    
    custom_features_resampled = Aggregator.group_between_dates(e.baseline, 
                                                            e.features, 
                                                            {col: 'sum' for col in e.features.columns[2:]})
    
    weather_features_resampled = Aggregator.group_between_dates(e.baseline, 
                                                             weather_features, 
                                                             Aggregator.weather_agg)
    
    # Merge features
    features = custom_features_resampled.merge(weather_features_resampled, left_index=True, right_index=True)
        
    return e.baseline, features

if __name__ == "__main__":
    path = r"C:\Users\nramirez\OneDrive - Enova Facilities Management\Desktop\SmartBaseline\Input_template.xlsx"
    b, f = preprocess_data(path)