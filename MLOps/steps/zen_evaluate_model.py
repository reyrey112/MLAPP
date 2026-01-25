import logging
import pandas as pd
from MLOps.src.evaluation_scores import *
from zenml import step
from sklearn.base import RegressorMixin
from typing import Tuple
from typing_extensions import Annotated
from zenml.client import Client
import mlflow
from zenml.experiment_trackers import BaseExperimentTracker

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(
    model, x_test: pd.DataFrame, y_test: pd.DataFrame
) -> Tuple[Annotated[float, "r2_score"], Annotated[float, "rmse"]]:
    try:
        prediction = model.predict(x_test)
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test, prediction)

        mlflow.log_metric("mse", mse)

        r2_class = R2()
        r2 = r2_class.calculate_score(y_test, prediction)

        mlflow.log_metric("r2", r2)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test, prediction)

        mlflow.log_metric("rmse", rmse)

        return r2, rmse

    except Exception as e:
        logging.error(f"Error during model evaluation: {e}")
        raise e
