from datetime import date

from sklearn.model_selection import train_test_split
from src.dataloader import GetRawData
from src.feature_engineering import FeatureEngineeringOnData
from src.modelbuilding import Models
from src.dataprep import PerformEda
from src import logger

from datetime import datetime

time = datetime.now().strftime("%d_%m_%Y")

class TrainModel:
    def Train(self,path):
        self.raw=GetRawData()
        self.data=self.raw.read_data_from_path(path)
        self.data=self.raw.rename_feature_columns()#returns the dataframe read from a file
        self.eda=PerformEda(self.data)
        self.X,self.y=self.eda.get_cleaned_data()
        self.feature_engineering=FeatureEngineeringOnData(self.X)
        self.df=self.feature_engineering.encode_categorical_variable_for_feature_class(['Type'])
        self.df=self.feature_engineering.standardize_the_data(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test=train_test_split(self.df,self.y,shuffle=True,stratify=self.y,random_state=120,test_size=0.3)
        #Model Building
        model_list=Models()
        self.final_list_models=model_list.compare_models(self.X_train, self.X_test, self.y_train, self.y_test)
        best_model=model_list.metrics(self.final_list_models)
        if len(self.y_train.value_counts())==2:
            model_list.plot_roc_curves(self.X_train, self.X_test, self.y_train, self.y_test)
        return best_model
