from django.apps import AppConfig
import logging
from dotenv import load_dotenv
import os


load_dotenv()



class ProcessingConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "processing"

    def ready(self):
        # from . import initializers

        # initializers.run_on_startup()
        # print("Initializer Running")

        from zenml.client import Client
        from zenml.enums import StackComponentType
        from zenml.exceptions import CredentialsNotValid, EntityExistsError

        print("initializing zenml client")
        client = Client()

        # try:
        #     print("checking if active stack exists")
        #     client.active_stack

        # except (
        #     CredentialsNotValid
        # ) as e:  # need to make something to make an acconut for user
        #     print("Active stack doesn't exist, looking for guest stack")
        #     # logging.debug(f"{e}")

        # try:
        #     stack = client.get_stack(name_id_or_prefix="guest_stack")
        #     if client.active_stack.experiment_tracker is None:
        #         print("no experiment tracker found, creating new guest stack")
        #         raise KeyError
        # except KeyError as e: #Add try except statment for all of the compoenentns
        print("no stack found, creating new guest stack")

        mlflow_port = os.environ.get("MLFLOW_PORT")
        s3_bucket = os.environ.get("AWS_STORAGE_BUCKET_NAME")
        try:
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

        except EntityExistsError as e:
            print("experiment tracker exists, continuing")

        try:
            client.create_stack_component(
                name="guest_model_registry",
                flavor="mlflow",
                component_type=StackComponentType.MODEL_REGISTRY,
                configuration={},
            )
        except EntityExistsError as e:
            print("model registry exists, continuing")

        try:
            client.create_stack_component(
                name="guest_deployer",
                flavor="docker",
                component_type=StackComponentType.DEPLOYER,
                configuration={},
            )
        except EntityExistsError as e:
            print("experiment tracker exists, continuing")

        try:
            client.create_stack_component(
                name="guest_artifact_store",
                flavor="s3",
                component_type=StackComponentType.ARTIFACT_STORE,
                configuration={"path": f"s3://{s3_bucket}"},
            )
        except EntityExistsError as e:
            print("artifact store exists, continuing")

        try:
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
        except EntityExistsError as e:
            print("stack exists, continuing")

        client.activate_stack("guest_stack")
