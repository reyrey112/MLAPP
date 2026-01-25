from zenml import pipeline
from MLOps.steps.zen_ingest_data import ingest_df
from MLOps.steps.zen_clean_data import clean_df
from MLOps.steps.zen_model_train import zen_train_model
from MLOps.steps.zen_evaluate_model import evaluate_model
from zenml_helper import zenml_parse, pydantic_model
from MLOps.steps.zen_config import ModelNameConfig
from zenml.config import DockerSettings
from zenml.config.docker_settings import DockerBuildConfig
from zenml.orchestrators.local_docker.local_docker_orchestrator import (
    LocalDockerOrchestratorSettings,
)
import mlflow

custom_build = DockerBuildConfig(
    dockerignore="/mnt/c/Users/reyde/Desktop/Coding_Project/Portfolio/MLapp/.dockerignore"
)
# docker_settings = DockerSettings(dockerfile="Dockerfile", dockerignore=".dockerignore", build_context_root=".")
docker_settings = DockerSettings(
    parent_image_build_config=custom_build, apt_packages=["build-essential"]
)
orc = LocalDockerOrchestratorSettings(
    run_args={
        "volumes": {
            "/home/reyde/Linux_mlapp/MLapp/user_files/csvs": {
                "bind": "/home/reyde/Linux_mlapp/MLapp/user_files/csvs",
                "mode": "rw",
            }
        }
    }
)


@pipeline(enable_cache=False, settings={"docker": docker_settings, "orchestrator": orc})
def train_pipeline(zenml_help: pydantic_model):
    """
    Docstring for train_pipeline

    :param data_path: Description
    :type data_path: str
    """

    df = ingest_df(zenml_help.zenml_data.data_path)
    x_train, x_test, y_train, y_test = clean_df(
        df=df,
        dropped_columns=zenml_help.zenml_data.dropped_columns,
        y_variable=zenml_help.zenml_data.y_variable,
        random_state=zenml_help.zenml_data.random_state,
        scaler=zenml_help.zenml_data.transformations,
        outliers=zenml_help.zenml_data.outliers,
    )
    model = zen_train_model(
        x_train, x_test, y_train, y_test, zenml_help.zenml_data.model_class
    )
    r2_score, rmse = evaluate_model(model, x_test, y_test)


# @pipeline(enable_cache=True)
# def existing_pipeline(zenml_help: pydantic_model):
#     """
#     Docstring for existing_pipeline

#     :param zenml_help: Description
#     :type zenml_help: pydantic_model
#     """
#     df = ingest_df(zenml_help.zenml_data.data_path)
