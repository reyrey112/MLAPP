import logging
import pandas as pd
from zenml import step
from MLOps.src.data_cleaning import *
from typing_extensions import Annotated
from typing import Tuple
from zenml_helper import pydantic_model


@step
def clean_df(
    df: pd.DataFrame, dropped_columns, y_variable, random_state, scaler, outliers
) -> Tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:
    """
    Docstring for clean_df

    :param df: Description
    :type df: pd.DataFrame
    :return: Description
    :rtype: Tuple[DataFrame, DataFrame, Series[Any], Series[Any]]
    """

    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data(
            dropped_columns, y_variable, random_state, scaler, outliers
        )

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        x_train, x_test, y_train, y_test = data_cleaning.handle_data(
            dropped_columns, y_variable, random_state, scaler, outliers
        )
        logging.info("Data cleaning compleate.")
        return x_train, x_test, y_train, y_test
    except Exception as e:
        logging.error(f"Error in data cleaning step: {e}")
        raise e
