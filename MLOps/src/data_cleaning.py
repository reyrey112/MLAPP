import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split
from zenml_helper import zenml_parse, pydantic_model
from model_predictions import model_predicting


class DataStrategy(ABC):
    """
    Docstring for DataStrategy

    Abstract Class defining strat for handing data
    """

    @abstractmethod
    def handle_data(
        self,
        data: pd.DataFrame,
        dropped_columns,
        y_variable,
        random_state,
        scaler,
        outliers,
    ) -> Union[pd.DataFrame, pd.Series]:

        pass


class DataPreProcessStrategy(DataStrategy):
    """
    Docstring for DataPreProcessStrategy

    Strategy for preprocessing Data
    """

    def handle_data(
        self,
        data: pd.DataFrame,
        dropped_columns,
        y_variable,
        random_state,
        scaler,
        outliers,
    ) -> pd.DataFrame:
        try:
            data = data.drop(
                dropped_columns,
                axis=1,
            )

            # make something to handle N/A values

            return data

        except Exception as e:
            logging.error(f"Error in DataPreProcessStrategy: {e}")
            raise e


class DataDivideStrategy(DataStrategy):
    """
    Docstring for DataDivideStrategy

    Strategy for dividing Data into training and testing sets
    """

    def handle_data(
        self,
        data: pd.DataFrame,
        dropped_columns,
        y_variable,
        random_state,
        scaler,
        outliers,
    ) -> Union[pd.DataFrame, pd.Series]:
        models = model_predicting()
        try:

            X = data.drop(y_variable, axis=1)
            y = data[y_variable]

            if outliers == True:
                y = models.outlier_removal(y)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=random_state
            )

            if scaler in models.scalers:
                scale = models.scalers[f"{scaler}"]()
                x_train, x_test = models.data_scaling(X_train, X_test, scale)

            return X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"Error in DataDivideStrategy: {e}")
            raise e


class DataCleaning:
    """
    Class for cleaning data which processes and divides data into test and train
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(
        self, dropped_columns, y_variable, random_state, scaler, outliers
    ) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(
                self.data, dropped_columns, y_variable, random_state, scaler, outliers
            )
        except Exception as e:
            logging.error(f"Error in handling data: {e}")
            raise e


# if __name__ == "__main__":
#     data = pd.read_csv("olist_order_reviews_dataset.csv")
#     data_cleaning = DataCleaning(data, DataPreProcessStrategy())
#     data_cleaning.handle_data(zenml_help)
