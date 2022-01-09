from src import logger

import os
import pathlib
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import plot_roc_curve
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import class_weight
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from datetime import datetime
from sklearn.metrics import RocCurveDisplay
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
import seaborn as sns

time = datetime.now().strftime("%d_%m_%Y")

class Models:
    def __init__(self):
        self.model_path=str(pathlib.Path().resolve())+"\\models\\"
        self.save_folder=str(pathlib.Path().resolve())+"\\visualization"
        # first, initialize the classificators
        self.models = [
          ('LogReg', LogisticRegression(solver='liblinear')),
          ('RF', RandomForestClassifier()),
          ('KNN', KNeighborsClassifier()),
          ('SVM', SVC()),
          ('GNB', GaussianNB()),
          ('XGB', XGBClassifier(use_label_encoder=False))
        ]
        self.trained_models=[]
        self.dfs=[]
        self.results = []
        self.names = []
        self.scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc']
        self.target_names = ['No Failure', 'Failure']

    def compare_models(self,X_train,X_test,y_train,y_test):
        for name, model in self.models:
            kfold = model_selection.KFold(n_splits=5, shuffle=True, random_state=120)
            cv_results = model_selection.cross_validate(model, X_train, y_train, cv=kfold, scoring=self.scoring)
            clf = model.fit(X_train, y_train)
            self.trained_models.append(tuple((name,model)))
            pickle.dump(model, open(f"{self.model_path}{name}.pkl", 'wb'))
            # with open(f"{self.model_path}{name}.pkl", 'wb') as f:
            #     pickle.dump(object, f)
            y_pred = clf.predict(X_test)
            print(name)
            print(classification_report(y_test, y_pred, target_names=self.target_names))
            self.results.append(cv_results)
            self.names.append(name)
            this_df = pd.DataFrame(cv_results)
            this_df['model'] = name
            self.dfs.append(this_df)
        final = pd.concat(self.dfs, ignore_index=True)
        return final

    def metrics(self,final):
        bootstraps = []
        for model in list(set(final.model.values)):
            model_df = final.loc[final.model == model]
            bootstrap = model_df.sample(n=30, replace=True)
            bootstraps.append(bootstrap)
        bootstrap_df = pd.concat(bootstraps, ignore_index=True)
        results_long = pd.melt(bootstrap_df,id_vars=['model'],var_name='metrics', value_name='values')

        time_metrics = ['fit_time','score_time'] # fit time metrics

        ## PERFORMANCE METRICS
        results_long_nofit = results_long.loc[~results_long['metrics'].isin(time_metrics)] # get df without fit data
        results_long_nofit = results_long_nofit.sort_values(by='values')
        # results_long_nofit.to_csv(f"results.csv")
        ##Plot results
        plt.figure(figsize=(40, 12))
        sns.set(font_scale=2.5)
        g = sns.boxplot(x="model", y="values", hue="metrics", data=results_long_nofit, palette="Set3")
        plt.legend(bbox_to_anchor=(1, 1), loc=2)
        plt.title('Comparison of Model by Classification Metric')
        plt.savefig(f'{self.save_folder}\\benchmark_models_performance.png',dpi=300)
        return self.get_best_model(results_long_nofit)

    def get_best_model(self,df):
        top_model=df.groupby(['metrics']).max().reset_index()
        #As of now we are interested in Recall
        val=top_model.loc[top_model['metrics']=='test_recall_weighted']['model'].reset_index(drop=True)
        model_dir=self.model_path
        for model in os.listdir(model_dir):
            if model.startswith(val.iloc[0]):
                return model_dir+f"{model}"


    def plot_roc_curves(self,X_train,X_test,y_train,y_test):
        axis=''
        for name,model in self.trained_models:
            if axis=='':
                fig = plot_roc_curve( model, X_test, y_test)
                axis=fig.ax_
            else:
                fig = plot_roc_curve( model, X_test, y_test, ax = axis)
            fig.figure_.suptitle("ROC curve comparison")
        plt.show()


        #     fig, ax = plt.subplots()
        #     RocCurveDisplay.from_estimator(
        #     model, X_test, y_test, ax = ax)
        #     if type(model)==RandomForestClassifier or type(model)==KNeighborsClassifier or type(model)==GaussianNB or type(model)==XGBClassifier:
        #         rfc_y_pred_proba = model.predict_proba(X_test)
        #         rfc_y_pred_proba_positive = rfc_y_pred_proba[:, 1]
        #         RocCurveDisplay.from_estimator(
        #         model, X_test, y_test, ax = ax)
        #         metrics.RocCurveDisplay.from_predictions(y_test,rfc_y_pred_proba_positive,ax=ax,name="rfc")
        #     else:
        #         model_decision = model.decision_function(X_test)
        #         metrics.RocCurveDisplay.from_predictions(y_test,model_decision,ax=ax,name=f"{name} predictions")

        # plt.show()
