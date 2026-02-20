import pickle
from zenml import step
from zenml.client import Client
import mlflow
client = Client()

experiment_tracker = client.active_stack.experiment_tracker
@step(experiment_tracker=experiment_tracker.name)
def log_model(model_path: str, run_name: str):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    mlflow.sklearn.log_model(model, name=run_name)

    return model
