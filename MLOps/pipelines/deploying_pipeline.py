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


# @step
def get_model(
    model_name: str,
) -> Tuple[Annotated[RegisteredModel, "model"], Annotated[str, "version"]]:
    client = Client()
    model_registry: MLFlowModelRegistry
    model_registry = client.active_stack.model_registry
    model = model_registry.get_model(model_name)
    version = model_registry.get_latest_model_version(model_name).version
    return model, version


# @step
def get_run_name(model_name: str):
    logging.warning("start get run name")

    mlflow_client = MlflowClient()
    client = Client()
    model_registry: MLFlowModelRegistry
    logging.warning("getting active model registry")

    model_registry = client.active_stack.model_registry

    logging.warning(f"{model_registry.list_models()}")

    logging.warning("getting latest model version")

    version = model_registry.get_latest_model_version(model_name).version
    logging.warning("getting registered model")

    registered_model = mlflow_client.get_model_version(model_name, version)
    run_name = registered_model.tags["zenml_run_name"]
    logging.warning("end get run name")

    return run_name


# @step
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


docker_settings = DockerSettings(required_integrations=[MLFLOW])
import os


@pipeline(
    enable_cache=False,
    settings={"docker": docker_settings},
)
def deploy_pipeline(
    zenml_help: pydantic_model,
    min_accuracy: float = 0,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):

    model, version = get_model(zenml_help.zenml_data.registered_model_name)
    # run_name = get_run_name(zenml_help.zenml_data.registered_model_name)
    get_prediction_model(zenml_help.zenml_data.run_name)
    mlflow_model_deployer_step(
        model=model,
        experiment_name="train_pipeline",
        run_name=zenml_help.zenml_data.run_name,
        workers=workers,
        timeout=timeout,
    )


def init_model(registered_model_name: str = "None", run_name: str = "None"):
    logging.warning("start init model")
    logging.warning(f"{registered_model_name}")

    loaded_model = get_prediction_model(run_name)
    logging.warning("complete init model")

    return loaded_model


@step
def model_predictions(x_dict: dict):
    model = get_step_context().pipeline_state

    x = pd.DataFrame(x_dict, index=[0])

    logging.warning(x)
    from sklearn.linear_model import LinearRegression

    model: LinearRegression
    y_pred = model.predict(x).tolist()

    return y_pred


root_path = Client().active_stack.artifact_store.path
docker_settings = DockerSettings(apt_packages=["build-essential"])
deployer_settings = DockerDeployerSettings(run_args={"network": "mlapp_default"})

orc = LocalDockerOrchestratorSettings(
    run_args={
        "network": "mlapp_default",
        # "volumes": {
        #     f"/home/ubuntu/MLAPP": {
        #         "bind": f"/app",
        #         "mode": "rw",
        #     }
        # },
    }
)


@pipeline(
    enable_cache=False,
    on_init=init_model,
    settings={
        "deployer": deployer_settings,
        "docker": docker_settings,
        "orchestrator": orc,
    },
)
def formulation_pipeline(
    Mixing_Time: float = 0.0,
    Active_1: float = 0.0,
    Active_2: float = 0.0,
    RM_3: float = 0.0,
    RM_4: float = 0.0,
    Active_3: float = 0.0,
    RM_6: float = 0.0,
    RM_7: float = 0.0,
    RM_8: float = 0.0,
    Active_4: float = 0.0,
    Water: float = 0.0,
    Crashout: float = 0.0,
):
    x_dict = {
        "Mixing_Time": Mixing_Time,
        "Active_1": Active_1,
        "Active_2": Active_2,
        "RM_3": RM_3,
        "RM_4": RM_4,
        "Active_3": Active_3,
        "RM_6": RM_6,
        "RM_7": RM_7,
        "RM_8": RM_8,
        "Active_4": Active_4,
        "Water": Water,
        "Crashout": Crashout,
    }
    return model_predictions(x_dict)
