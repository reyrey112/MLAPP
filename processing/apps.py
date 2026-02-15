from django.apps import AppConfig
import logging
import os


class ProcessingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "processing"

    def ready(self):
        # from . import initializers

        # initializers.run_on_startup()
        # print("Initializer Running")

        from zenml.client import Client
        from zenml.enums import StackComponentType
        from zenml.exceptions import CredentialsNotValid

        client = Client()

        try:

            client.active_stack

        except (
            CredentialsNotValid
        ) as e:  # need to make something to make an acconut for user
            print("Creating Default Setup")
            # logging.debug(f"{e}")

        if client.active_stack.experiment_tracker is None:
            print("No experiment tracker, looking for guest stack")

            try:

                stack = client.get_stack(name_id_or_prefix="guest_stack")

            except KeyError as e: #Add try except statment for all of the compoenentns
                print("no stack found, creating new guest stack")

                mlflow_port = os.environ.get("MLFLOW_PORT")
                s3_bucket = os.environ.get("AWS_STORAGE_BUCKET_NAME")

                client.create_stack_component(
                    name="guest_tracker",
                    flavor="mlflow",
                    component_type=StackComponentType.EXPERIMENT_TRACKER,
                    configuration={
                        "tracking_uri": f"http://127.0.0.1:{mlflow_port}",
                        "tracking_username": "guest",
                        "tracking_password": "Guest1.",
                    },
                )
                
                client.create_stack_component(
                    name="guest_model_registry",
                    flavor="mlflow",
                    component_type=StackComponentType.MODEL_REGISTRY,
                    configuration={},
                )

                client.create_stack_component(
                    name="guest_deployer",
                    flavor="docker",
                    component_type=StackComponentType.DEPLOYER,
                    configuration={},
                )

                client.create_stack_component(
                    name="guest_artifact_store",
                    flavor="s3",
                    component_type=StackComponentType.ARTIFACT_STORE,
                    configuration={"path": f"s3://{s3_bucket}"},
                )

                client.create_stack(
                    name="guest_stack",
                    components={
                        StackComponentType.ORCHESTRATOR: "default",
                        StackComponentType.ARTIFACT_STORE: "guest_artifact_store",
                        StackComponentType.EXPERIMENT_TRACKER: "guest_tracker",
                        StackComponentType.MODEL_REGISTRY: "guest_model_registry",
                        StackComponentType.DEPLOYER: "guest_deployer",
                    },
                )

                client.activate_stack("guest_stack")
