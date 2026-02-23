import pickle
from zenml import step
from zenml.client import Client
import mlflow
from zenml.integrations.mlflow.experiment_trackers.mlflow_experiment_tracker import (
    MLFlowExperimentTracker,
)

client = Client()
experiment_tracker: MLFlowExperimentTracker

experiment_tracker = client.active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def log_model(model_path: str, run_name: str):
    mlflow.set_tracking_uri(experiment_tracker.get_tracking_uri)
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    mlflow.sklearn.log_model(model, name=run_name)

    return model
