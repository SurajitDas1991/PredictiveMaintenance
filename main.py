import shutil
from src.dashboard import PrepareTheDasboard
from src import logger
from src.train import TrainModel
from src.predict import PredictFromModel

import pandas as pd
from datetime import datetime
import pathlib
import os

time=datetime.now().strftime("%d_%m_%Y")
app_logger=logger.get_logger(__name__,f"StartOfProgram.txt_{time}")

trained_model=None

def train_the_model():
    path=str(pathlib.Path().resolve())+"\\data\\predictiveMaintenance_26122021_031456.csv"
    global trained_model
    trained_model=TrainModel().Train(path)
    if os.path.exists('best_model'):
        os.removedirs('best_model')
        os.makedirs('best_model')
        shutil.copy(f'{trained_model}','best_model')
    else:
        os.makedirs('best_model')
        shutil.copy(f'models/{trained_model}','best_model')

if __name__=='__main__':

    # Streamlit dashboard
    dashboard_prep= PrepareTheDasboard()
    dashboard_prep.get_ready()
    #Get the Data and place in a database
    #Start with EDA process
    #Start with Feature Engineering
    #train_the_model()
    #Start with Model Building
    #Start with Deployment
    #predict_from_model()
