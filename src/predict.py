from sklearn.model_selection import train_test_split
from src.dataloader import GetRawData
from src.feature_engineering import FeatureEngineeringOnData
from src.modelbuilding import Models
from src.dataprep import PerformEda
from src import logger

import pickle
from datetime import datetime

time = datetime.now().strftime("%d_%m_%Y")

class PredictFromModel:
    def set_up_dataframe(self,dict_of_data):
        pass

    def predict(self,df,model):
        self.raw=GetRawData()
        self.raw.set_data_for_prediction(df)
        self.data=self.raw.rename_feature_columns()#returns the dataframe read from a file
        self.eda=PerformEda(self.data)
        self.X=self.data
        #self.X=self.eda.get_cleaned_data_for_prediction()
        self.feature_engineering=FeatureEngineeringOnData(self.X)
        self.df=self.feature_engineering.encode_categorical_variable_for_feature_class(['Type'])
        self.df=self.feature_engineering.standardize_the_data(self.X)
        #Start prediction
        loaded_model=None
        loaded_model=pickle.load(open(model,'rb'))
        # with open(model,'rb') as modelFile:
        #     loaded_model = pickle.load(modelFile)
        #print(type(loaded_model))
        prediction=loaded_model.predict(self.df)
        return prediction
