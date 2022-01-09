import pandas as pd
import sqlite3 as sql
from src import logger

from src.dataprep import PerformEda


class GetRawData:
    def set_data_for_prediction(self,df):
        self.data=df

    def read_data_from_path(self, path):
        self.data = pd.read_csv(path)
        return self.data
        #self.cleaned_data = PerformEda(self.data).get_cleaned_data()
        #self.write_to_db(self.cleaned_data)
        #return self.cleaned_data

    def rename_feature_columns(self):
        self.data.rename(columns = {'Air temperature [K]':'Air temperature', 'Process temperature [K]':'Process temperature',
                              'Rotational speed [rpm]':'Rotational speed',
                         'Torque [Nm]':'Torque','Tool wear [min]':'Tool wear'},inplace = True)
        return self.data

    def write_to_db(self, dataframe):
        conn = sql.connect('../database/weather.db')
        dataframe.to_sql('weather', conn)
