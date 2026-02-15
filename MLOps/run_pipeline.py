from MLOps.pipelines.training_pipeline import train_pipeline
from MLOps.pipelines.registering_pipeline import register_pipeline
from MLOps.pipelines.deploying_pipeline import formulation_pipeline
from MLOps.pipelines.batch_inference_pipeline import batch_formulation_pipeline

# from MLOps.pipelines.deploying_pipeline import deploy_pipeline
# from MLOps.pipelines.deployment_pipeline import continous_deployment_pipeline
from zenml_helper import zenml_parse, pydantic_model
from zenml.client import Client
import logging
import datetime
from mlflow import MlflowClient
import mlflow
from zenml import Model


# from zenml.integrations.mlflow.experiment_trackers.mlflow_experiment_tracker import MLFlowExperimentTracker

# mlflow.set_tracking_uri("postgresql://user:password@localhost:5431/mlflowdb")



def run(
    pipeline: str, zenml_help: pydantic_model = None, deployment_name: str = "None", file_path : str = 'none'
):
    uri = ""
    if pipeline == "train":
        train_pipe = train_pipeline.with_options(
            run_name=f"{zenml_help.zenml_data.model_name} {zenml_help.zenml_data.model_class} {datetime.datetime.now().replace(microsecond=0)}"
        )
        train_pipe(zenml_help)
        return uri

    if pipeline == "register":

        register_pipeline(zenml_help)
        return uri


    if pipeline == 'batch':
        batch_formulation_pipeline(zenml_help, file_path)
        return uri

    if pipeline == "deploy":
        formulation_pipeline.with_options(
            on_init_kwargs={
                "registered_model_name": zenml_help.zenml_data.registered_model_name,
                "run_name": zenml_help.zenml_data.run_name,
            },
            
        ).deploy(deployment_name=deployment_name)
    return uri

    