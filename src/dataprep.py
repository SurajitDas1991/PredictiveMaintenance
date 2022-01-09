import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, MissingIndicator
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from datetime import datetime
from imblearn.over_sampling import SMOTE, RandomOverSampler

from src.feature_engineering import FeatureEngineeringOnData
from src import logger

time = datetime.now().strftime("%d_%m_%Y")

class PerformEda:

    def __init__(self, dataframe):
        self.df = dataframe
        self.unwanted_column_list=['Failure Type','UDI','Product ID']
        self.logger = logger.get_logger(__name__, f"EDA.txt_{time}")

    def get_cleaned_data(self):
        '''
        Analyzes and preprocesses the data and makes it ready for feature engineering
        '''
        self.df=self.drop_unnecessary_columns(self.df,self.unwanted_column_list)
        X,y=self.separate_label_feature(self.df,"Target")
        X,y=self.handle_imbalanced_dataset(X,y)
        return X,y

    def get_cleaned_data_for_prediction(self):
        self.df=self.drop_unnecessary_columns(self.df,self.unwanted_column_list)
        return self.df

    def handle_imbalanced_dataset(self, X, y):
        sample = RandomOverSampler(sampling_strategy='minority')

        X_resampled ,y_resampled= sample.fit_resample(X, y)

        return X_resampled, y_resampled

    def drop_unnecessary_columns(self, data, column_name_list):
        """
        Method Name: is_null_present
        Description: This method drops the unwanted columns as discussed in EDA section.
        """
        data.drop(column_name_list, axis=1,inplace=True)
        return data

    def separate_label_feature(self, data, label_column_name):
        """
        Description: This method separates the features and a Label Columns.
        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
        """
        try:
            self.X = data.drop(
                labels=label_column_name, axis=1
            )  # drop the columns specified and separate the feature columns
            self.Y = data[label_column_name]  # Filter the Label columns
            return self.X, self.Y
        except Exception as e:
            self.logger.exception(
                "Exception occured in separate_label_feature method of the EDA class. Exception message:  "
                + str(e)
            )
            raise Exception()
