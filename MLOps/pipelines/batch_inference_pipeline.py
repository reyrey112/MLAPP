from zenml import pipeline, step, get_step_context, log_metadata
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.client import Client
from zenml.integrations.mlflow.model_registries.mlflow_model_registry import (
    MLFlowModelRegistry,
)
from zenml_helper import pydantic_model, zenml_parse
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from mlflow.tracking import MlflowClient
from typing import Tuple, Annotated
from zenml.model_registries.base_model_registry import RegisteredModel
from zenml.deployers.docker.docker_deployer import DockerDeployerSettings
from sklearn.svm import SVR
import pandas as pd

from zenml.orchestrators.local_docker.local_docker_orchestrator import (
    LocalDockerOrchestratorSettings,
)
import logging
from zenml.enums import StackComponentType
from prediction_powerBI_database_util import write_to_DB
from sklearn.base import RegressorMixin
from typing import Tuple, Any
import numpy as np

from dotenv import load_dotenv
import os


load_dotenv()

@step
def load_data(file_path: str):
    data = pd.read_csv(file_path)
    return data


@step
def get_prediction_model(run_name: str):
    logging.warning("start get prediction model")

    client = Client()

    pipeline = client.get_pipeline("train_pipeline")
    run = pipeline.get_runs(name=run_name)[0]
    train_step = run.steps["zen_train_model"]
    model_artifact = train_step.outputs["output"][0]

    model = model_artifact.load()
    logging.warning("end get prediction model")
    return model


@step
def model_predictions(
    model, data: pd.DataFrame
) -> Tuple[Annotated[np.ndarray, "predictions"], Annotated[pd.DataFrame, "data"]]:
    from sklearn.linear_model import LinearRegression

    predictions = model.predict(data)

    data["viscosity"] = predictions

    return predictions, data


docker_settings = DockerSettings(
    apt_packages=["build-essential"], requirements=["psycopg2-binary"]
)
orc = LocalDockerOrchestratorSettings(
    run_args={
        "volumes": {
            "/tmp": {
                "bind": "/tmp",
                "mode": "rw",
            }
        }
    }
)


@step
def write_to_database(db_name, user, password, host, port, data, table_name):
    write_to_DB(
        db_name=db_name,
        user=user,
        password=password,
        host=host,
        port=port,
        table_name=table_name,
        data=data,
    )


@pipeline(enable_cache=False, settings={"docker": docker_settings, "orchestrator": orc})
def batch_formulation_pipeline(zenml_help: pydantic_model, file_path: str):
    logging.warning(file_path)
    data = load_data(file_path)
    model = get_prediction_model(zenml_help.zenml_data.run_name)
    predictions, data_new = model_predictions(model=model, data=data)
    write_to_database(
        db_name="powerbi_prod_tables",
        user="postgres",
        password="password",
        host=os.environ.get("POSTGRES_DB_HOST"),
        port=os.environ.get("POSTGRES_DB_POST"),
        table_name="formulation_table",
        data=data_new,
    )
    return predictions
