import streamlit as st
from src.predict import PredictFromModel
import pandas as pd
import pathlib
import os

class PrepareTheDasboard:
    def get_ready(self):
        st.header("PREDICTIVE MAINTENANCE - FAILURE OR NOT")
        st.text("Enter the values below - All the fields except the first dropdown \ntake in numerical values")
        self.selected_option=st.selectbox("Type of Machine :",('H','L','M'))
        self.air_temperature=st.text_input("Air temperature [K] :")
        self.process_temperature=st.text_input("Process temperature [K] :")
        self.rotational_speed=st.text_input("Rotational speed [rpm] :")
        self.torque=st.text_input("Torque [Nm] :")
        self.tool_wear=st.text_input("Tool wear [min] :")
        best_model=''
        if st.button("Predict Failure "):
            path_dir=os.path.join(f"{str(pathlib.Path().resolve())}", "best_model", "")
            #path_dir=str(pathlib.Path().resolve())+"\\best_model\\"
            for f in os.listdir(path_dir):
                best_model=f
            #Check for valdid values
            if self.air_temperature=='' or  self.process_temperature=='' or self.rotational_speed=='' or self.torque=='' or self.tool_wear==0:
                st.text("Please provide appropriate values in the boxes")
                return
            try:
                st.text("Preparing the data for prediction ...!!")
                dict_for_pred=  {
                "Type":[str(self.selected_option)],
                "Air temperature [K]":[float(self.air_temperature)],
                "Process temperature [K]":[float(self.process_temperature)],
                "Rotational speed [rpm]":[float(self.rotational_speed)],
                "Torque [Nm]":[float(self.torque)],
                "Tool wear [min]":[float(self.tool_wear)]}
                df=pd.DataFrame(data=dict_for_pred)
                pred=PredictFromModel()
                model=f"{path_dir}{best_model}"
                actual_prediction=pred.predict(df,model)
                if actual_prediction==1:
                    st.text("Probability of failure is high !")
                else:
                    st.text("No probability of failure")
            except Exception  as e:
                st.text(f"Error in predicting the model {str(e)}")
