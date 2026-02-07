from django.apps import AppConfig
import socket, os


class ProcessingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "processing"

    def ready(self):
        # from . import initializers

        # initializers.run_on_startup()
        # print("Initializer Running")
        from zenml.enums import StackComponentType
        from zenml.client import Client
        import mlflow
        from zenml.integrations.mlflow.experiment_trackers.mlflow_experiment_tracker import MLFlowExperimentTracker
        import logging
        client = Client()

        if os.environ.get("LOCAL_DEV_CLOUD") == "DEV":
            client.activate_stack("local_testing")
            print(f"ZenML stack set to: {client.active_stack.name}")

            mlflow_host = os.environ.get("MLFLOW_HOST", "mlflow_service")

            try:
                mlflowip = socket.gethostbyname(mlflow_host)
                tracking_uri = f"http://{mlflowip}:5000"
                print(f"tracking_uri: {tracking_uri}")
            except socket.gaierror:
                print(f"Could not resolve {mlflow_host}, using hostname")
                tracking_uri = f"http://{mlflow_host}:5000"

            mlflow.set_tracking_uri(tracking_uri)

            try:
                tracker_name = client.active_stack_model.components.get(
                    StackComponentType.EXPERIMENT_TRACKER
                )

                client.update_stack_component(
                    name_id_or_prefix=tracker_name[0].name,
                    component_type=StackComponentType.EXPERIMENT_TRACKER,
                    configuration={
                        "tracking_uri": tracking_uri,
                        "tracking_username": os.environ.get("MLFLOW_TRACKING_USERNAME"),
                        "tracking_password": os.environ.get("MLFLOW_TRACKING_PASSWORD"),
                    },
                )
                print(f"componenet updated with tracking uri")

            except Exception as e:
                print(f"Could not update existing tracker: {e}")
                print("Creating new experiment tracker")

                try:
                    from zenml.integrations.mlflow.experiment_trackers import (
                        MLFlowExperimentTracker,
                    )

                    # Register the tracker
                    client.create_stack_component(
                        name=os.environ.get("MLFLOW_EXPERIMENT_NAME"),
                        component_type=StackComponentType.EXPERIMENT_TRACKER,
                        flavor="mlflow",
                        configuration={
                            "tracking_uri": tracking_uri,
                            "tracking_username": os.environ.get(
                                "MLFLOW_TRACKING_USERNAME"
                            ),
                            "tracking_password": os.environ.get(
                                "MLFLOW_TRACKING_PASSWORD"
                            ),
                        },
                    )
                    print("Created new MLflow experiment tracker")
                except Exception as e:
                    print(f"Could not create tracker: {e}")
                    raise TypeError
        elif os.environ.get("LOCAL_DEV_CLOUD") == "CLOUD":
            pass
        else:
            from mlflow import MlflowClient
            client.activate_stack("local_stack")
            experiment_tracker = client.active_stack.experiment_tracker
        
            print(f"ZenML stack set to: {client.active_stack.name}")
