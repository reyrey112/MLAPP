from zenml.client import Client
import mlflow
from zenml import pipeline, step
import logging
import pandas as pd
from zenml_helper import pydantic_model, zenml_parse
from zenml.integrations.mlflow.steps.mlflow_registry import mlflow_register_model_step
from typing_extensions import Annotated
import typing
from sklearn.base import RegressorMixin
from zenml.integrations.mlflow.model_registries.mlflow_model_registry import (
    MLFlowModelRegistry,
)
from zenml.integrations.mlflow.experiment_trackers.mlflow_experiment_tracker import (
    MLFlowExperimentTracker,
)
from zenml.config import DockerSettings
import os


@step(enable_cache=False)
def get_model(
    run_name: str,
) -> typing.Tuple[Annotated[typing.Any, "model"], Annotated[str, "uri"]]:
    client = Client()

    pipeline = client.get_pipeline("train_pipeline")
    runs = pipeline.get_runs(name=run_name)
    run = runs[0]
    run_id = str(run.id)

    # train_step = run.steps["zen_train_model"]
    train_step = run.steps["log_model"]

    model_artifact = train_step.outputs["output"][0]

    model = model_artifact.load()
    uri = model_artifact.uri
    logging.warning(f"{uri}")
    # logging.warning(f"file://{os.path.abspath(uri)}")

    return model, uri

docker_settings = DockerSettings(apt_packages=["build-essential"])
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def register_pipeline(zenml_help: pydantic_model):

    model, uri = get_model(run_name=zenml_help.zenml_data.run_name)
    mlflow_register_model_step(
        model=model,
        name=(
            zenml_help.zenml_data.model_name
            if zenml_help.zenml_data.registered_model_name == "None"
            else zenml_help.zenml_data.registered_model_name
        ),
        # model_source_uri = uri,
        run_name=zenml_help.zenml_data.run_name,
        experiment_name="train_pipeline",
    )
