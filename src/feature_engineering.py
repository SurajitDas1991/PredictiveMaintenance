import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, MissingIndicator
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import logging
from datetime import datetime

# Custom libraries
from src import logger

time = datetime.now().strftime("%d_%m_%Y")


class FeatureEngineeringOnData:
    def __init__(self, X):
        self.df = X
        self.logger = logger.get_logger(__name__, f"FeatureEngineering.txt_{time}")

    def check_for_null_values(self) -> pd.Series:
        return self.df.isnull().sum()

    def numerical_data(self) -> pd.DataFrame:
        return self.df.select_dtypes(include="O")

    def categorical_data(self) -> pd.DataFrame:
        return self.df.select_dtypes(exclude="O")

    def encode_categorical_variable_for_output_class(self, column_names):
        self.label_encoder = preprocessing.LabelEncoder()
        for column in column_names:
            self.df[column] = self.label_encoder.fit_transform(
                self.df[column]
            )
        return self.df

    def encode_categorical_variable_for_feature_class(self, column_names):
        self.ordinal_encoder = preprocessing.OrdinalEncoder()
        for column in column_names:
            self.df[[column]] = self.ordinal_encoder.fit_transform(
                self.df[[column]]
            )
        return self.df

    def standardize_the_data(self, data):
        self.data = data
        self.scaler = StandardScaler()
        self.scaled_data = self.scaler.fit_transform(self.data)
        self.scaled_num_df = pd.DataFrame(
            data=self.scaled_data, columns=self.df.columns, index=self.df.index
        )
        return self.scaled_num_df

    # def handle_imbalanced_dataset(self, X, y):
    #     sample = RandomOverSampler()

    #     X_resampled, y_resampled = sample.fit_resample(X, y)

    #     return X_resampled, y_resampled


    def replace_invalid_values_with_null(self, data):
        """
        Description: This method replaces invalid values i.e. '?' with null
        """
        data.replace("na", np.NaN, inplace=True)
        return data

    def drop_high_correlated_features(self, data):
        corr_matrix = data.corr().abs()
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Find index of feature columns with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        # Drop features
        data.drop(data[to_drop], axis=1, inplace=True)
        return data

    # def separate_label_feature(self, data: pd.DataFrame, label_column_name):
    #     """
    #     Description: This method separates the features and a Label Columns.
    #     Output: Returns two separate Dataframes, one containing features and the other containing Labels .
    #     """
    #     try:
    #         self.X = data.drop(
    #             labels=label_column_name, axis=1
    #         )  # drop the columns specified and separate the feature columns
    #         self.Y = data[label_column_name]  # Filter the Label columns
    #         return self.X, self.Y
    #     except Exception as e:
    #         self.logger.exception(
    #             "Exception occured in separate_label_feature method of the Preprocessor class. Exception message:  "
    #             + str(e)
    #         )
    #         self.logger.exception(
    #             "Label Separation Unsuccessful. Exited the separate_label_feature method of the Preprocessor class"
    #         )
    #         raise Exception()

    def plot_corr(self, data, size=10):
        """Function plots a graphical correlation matrix for each pair of columns in the dataframe.
        Input:
            data: pandas DataFrame
            size: vertical and horizontal size of the plot
        """
        corr = data.corr()
        fig, ax = plt.subplots(figsize=(size, size))
        ax.matshow(corr)
        plt.xticks(range(len(corr.columns)), corr.columns)
        plt.yticks(range(len(corr.columns)), corr.columns)

    def impute_data(self, data, impute_strategy: str):
        imp = SimpleImputer(missing_values=np.nan, strategy=impute_strategy)
        imp.fit(data)
        data_imputed = pd.DataFrame(imp.transform(data), columns=data.columns)
        return data_imputed

    def pca_fn(self, data: pd.DataFrame, number_of_components: int):
        pca = PCA(n_components=number_of_components)
        # prepare transform on dataset
        pca.fit(data)
        # apply transform to dataset
        transformed = pca.transform(data)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        sns.lineplot(data=pca.explained_variance_ratio_)
        plt.xlabel("Number of components")
        plt.ylabel("explained variance ratio")
        plt.title("Plot of number of components v/s explained variance ratio")
        plt.subplot(1, 2, 2)
        sns.lineplot(data=np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel("Number of components")
        plt.ylabel("Cumulative explained variance")
        plt.title("Plot of number of components v/s cumulative explained variance")
        plt.show()
        print(
            f"With Number of components as {number_of_components}, the total explained variance is ",
            pca.explained_variance_ratio_[:{number_of_components}].sum(),
        )
        return transformed
