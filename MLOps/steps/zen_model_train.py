import logging
import pandas as pd
from zenml import step
from MLOps.src.model_dev import *
from sklearn.base import RegressorMixin
from MLOps.steps.zen_config import ModelNameConfig
import mlflow
from zenml.client import Client
from model_predictions import model_predicting
from sklearn.svm import LinearSVC, LinearSVR, SVR, SVC
from sklearn.linear_model import LassoCV, ElasticNetCV, Ridge, SGDRegressor, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# from zenml.integrations.mlflow.experiment_trackers.mlflow_experiment_tracker import MLFlowExperimentTracker

experiment_tracker = Client().active_stack.experiment_tracker

regressor_dict = {
    "Lasso": LassoCV,
    "ElasticNet": ElasticNetCV,
    "Ridge Regression": Ridge,
    "Linear SVR": LinearSVR,
    "rbf SVR": SVR,
    "SGD Regressor": SGDRegressor
}

@step(experiment_tracker=experiment_tracker.name)
def zen_train_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_class: str,
) -> RegressorMixin:
    """
    Train a model using the provided training data.

    Args:
        x_train (pd.DataFrame): Training input features.
        x_test (pd.DataFrame): Testing input features.
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Testing target variable.

    Returns:
        RegressorMixin: Trained model.
    """

    model = None
    config = ModelNameConfig()
    models = model_predicting()
    try:
        if model_class in regressor_dict:
            mlflow.sklearn.autolog()
            trained_model, _, _ = models.train_regressor(model_class,x_train, x_test, y_train, y_test)
            return trained_model
        else:
            raise ValueError(f"Model {config.model_name} not supported")
    except Exception as e:
        logging.error(f"Error during model training: {e}")
        raise e
