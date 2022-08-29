import openpyxl
import pandas as pd
import numpy as np

class ReadExcel:
    
    def __init__(self, file_path: str):
        
        # Open Workbook
        self.wb = openpyxl.load_workbook(filename = file_path, data_only = True)
        sheet = self.wb["Project"]
        
        # Extract params from excel file
        self.project_name = sheet["C3"].value
        self.start = sheet["C4"].value
        self.end = sheet["C5"].value
        
        self.lat = sheet["C9"].value
        self.lon = sheet["C10"].value
        self.alt = sheet["C11"].value
        
        
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
    