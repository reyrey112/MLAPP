from django.shortcuts import render, redirect, get_object_or_404, get_list_or_404
from django.http import HttpResponse, HttpRequest
from django.core.files.images import ImageFile
from rest_framework import generics
from .models import *
import pandas as pd
import numpy as np
import sklearn
from model_predictions import model_predicting
import pickle
import matplotlib
from zenml.client import Client
import mlflow
from zenml.integrations.mlflow.experiment_trackers.mlflow_experiment_tracker import (
    MLFlowExperimentTracker,
)
from zenml.integrations.mlflow.model_registries.mlflow_model_registry import (
    MLFlowModelRegistry,
)
from mlflow import MlflowClient
# from zenml.deployers.docker.docker_deployer import DockerDeployer
import requests
import tempfile
import csv
from django.contrib import messages

matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
from MLapp.settings import MEDIA_URL, MEDIA_ROOT
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.http import Http404
import logging
import os, io, uuid
from django.contrib.auth import get_user_model, login
from zenml_helper import zenml_parse, pydantic_model
from MLOps.run_pipeline import run

from prediction_powerBI_database_util import create_database, create_tables
import json
from dotenv import load_dotenv
import os

User = get_user_model()

load_dotenv()


def register_as_guest(request: HttpRequest):
    if request.method == "GET":
        if request.user.is_anonymous:
            try:
                username = f"guest_{uuid.uuid4().hex[:12]}"
                guest_user = User.objects.create_user(username=username)
                guest_user.set_unusable_password()
                guest_user.profile.is_guest = True
                guest_user.save()

                # Get or create profile (in case signal didn't fire)
                # profile, created = UserProfile.objects.get_or_create(user=guest_user)
                # profile.is_guest = True
                # profile.save()

                # Log in the guest user
                login(request, guest_user)
                request.session["guest_user_id"] = guest_user.id

                # Redirect to next or home
                next_url = request.GET.get("next", "home")
                return redirect(next_url)
                
            except Exception as e:
                messages.error(request, f"Error creating guest account: {str(e)}")
                return redirect("login")
    
    return redirect("login")


def generate_guest_id():
    return str(uuid.uuid4())


def logon(request: HttpRequest):
    if request.user.is_authenticated:
        return redirect("/home/")
    return render(request, "login.html")


def create_user(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            
            # Profile should be created automatically by signal
            # But we can verify and set is_guest to False
            user.profile.is_guest = False
            user.profile.save()
            
            # Log the user in
            login(request, user)
            
            messages.success(request, f'Account created successfully!')
            return redirect('home')
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = UserCreationForm()
    
    return render(request, "register.html", {"form": form})


def redirect_home(request: HttpRequest):
    return redirect("/login/")


def redirect_end(request: HttpRequest, **kwargs):
    return render(request, "redirect.html", context=kwargs)


@login_required
def home(request: HttpRequest):
    # if request.user.is_authenticated:

    if hasattr(request.user, 'profile') and request.user.profile.is_guest:
        print("hello")

    return render(request, "home.html")
    # elif "guest_id" in request.session:
    #     return render(request, "home.html")
    # else:
    #     request.session["guest_id"] = generate_guest_id()
    #     return render(request, "home.html")

@login_required
def zenml_home(request: HttpRequest):
    return render(request, "zenml_home.html")

@login_required
def file_upload(request: HttpRequest):
    return render(request, "file_upload.html")

@login_required
def load_file(request: HttpRequest):

    try:
        file_list = get_list_or_404(
            upload_file,
            user=request.user,
        )

    except Http404 as e:
        return render(request, "load_file.html")

    viewing_table = pd.DataFrame()
    length = 0
    pks = []

    for file in file_list:
        pks.append(file.pk)
        columns = {"file": file.file.name[5:-4], "upload_time": file.upload_time}

        for column in columns:

            value = columns[column]
            viewing_table.loc[length, column] = value

        length = length + 1

    table_html = viewing_table.to_html(
        classes="table table-striped table-bordered", index=False
    )

    context = {"table": table_html}

    request.session["pks"] = pks

    return render(request, "load_file.html", context)

@login_required
def show_table(request: HttpRequest):
    if request.method == "POST":
        if "file" in request.FILES:
            try:
                user_file = request.FILES["file"]
                df = pd.read_csv(user_file)

                # Save to database
                saved_file = upload_file.objects.create(
                    file=user_file,
                    user=request.user
                )

                request.session["dk"] = saved_file.pk

                # Convert the DataFrame to an HTML table and pass it to the template.
                # Use classes for Bootstrap styling; in the template we mark it safe.
                table_html = df.to_html(
                    classes="table table-striped table-bordered", index=False
                )

                context = {"table": table_html}

                return render(request, "show_table.html", context)

            except Exception as e:
                return render(
                    request,
                    "show_table.html",
                    {
                        "exception": f"The following error occured: {e} \n Please upload another file"
                    },
                )
        elif "selected_file" in request.POST:
            try:
                file_indexes = [
                    int(index) for index in request.POST.getlist("selected_file")
                ]

                pks = request.session["pks"]

                file_pk = [pks[file_index] for file_index in file_indexes]

                file = get_object_or_404(upload_file, pk=file_pk[0])

                df = pd.read_csv(file.file)

                request.session["dk"] = file.pk

                table_html = df.to_html(
                    classes="table table-striped table-bordered", index=False
                )

                context = {"table": table_html}

                return render(request, "show_table.html", context)

            except Exception as e:
                return render(
                    request,
                    "show_table.html",
                    {
                        "exception": f"The following error occured: {e} \n Please upload another file"
                    },
                )
        else:
            return redirect("/redirect/", pk="pk")

    elif "dk" in request.session:
        df = pd.read_csv(
            get_object_or_404(upload_file, pk=request.session["dk"]).file
        )

        table_html = df.to_html(
            classes="table table-striped table-bordered", index=False
        )

        context = {"table": table_html}

        return render(request, "show_table.html", context)

    else:

        return redirect("/redirect/", pk="pk")

@login_required
def x_variable_selection(request: HttpRequest):
    if request.method == "GET":
        df = pd.read_csv(
            get_object_or_404(upload_file, pk=request.session["dk"]).file
        )

        context = {"options": df.columns.values}

        return render(request, "x_variable_selection.html", context)

    else:
        return redirect("/redirect/", pk="pk")

@login_required
def y_variable_selection(request: HttpRequest):
    if request.method == "POST":
        df = pd.read_csv(
            get_object_or_404(upload_file, pk=request.session["dk"]).file
        )

        columns_to_drop = request.POST.getlist("columns")
        df = df.drop(columns_to_drop, axis=1)

        request.session["columns_to_drop"] = columns_to_drop

        context = {"options": df.columns.values}

        return render(request, "y_variable_selection.html", context)

    else:
        return redirect("/redirect/", POST="POST")

@login_required
def model_selection(request: HttpRequest):
    if request.method == "POST":
        df = pd.read_csv(
            get_object_or_404(upload_file, pk=request.session["dk"]).file
        )
        y_variable = request.POST["dropdown"]
        request.session["y_variable"] = y_variable

        if request.session["columns_to_drop"] is not False:
            df = df.drop(request.session["columns_to_drop"], axis=1)

        y_column = df[y_variable]
        models = model_predicting()
        y_class = models.y_classifier(y_column)

        request.session["y_class"] = y_class

        column_num = len(df.columns)

        context = models.model_predict(y_column, column_num, y_class)

        context["options"] = models.supported_models

        return render(request, "model_selection.html", context)

    else:
        return redirect("/redirect/", POST="POST")

@login_required
def scaler_selection(request: HttpRequest):
    if request.method == "POST":
        file = get_object_or_404(upload_file, pk=request.session["dk"])
        df = pd.read_csv(file.file)
        request.session["model_class"] = request.POST["models"]
        request.session["model_name"] = request.POST["model_name"]

        y_class = request.session["y_class"]
        y_variable = request.session["y_variable"]

        models = model_predicting()

        if y_class == "regression":
            context = models.scaling_check(df[f"{y_variable}"])

            context["options"] = models.scalers

            return render(request, "scaler_selection.html", context)

        else:
            context = {"skip": "skip"}

            return render(request, "scaler_selection.html", context)

    else:
        return render(request, "redirect.html")

@login_required
def train_model(request: HttpRequest):
    if request.method == "POST":
        models = model_predicting()

        file = get_object_or_404(upload_file, pk=request.session["dk"])
        dropped_cols = request.session["columns_to_drop"]
        df = pd.read_csv(file.file)
        df = df.drop(dropped_cols, axis=1)
        y_variable = request.session["y_variable"]
        y_class = request.session["y_class"]
        model_class = request.session["model_class"]
        if request.POST["scalers"] == "no":
            scaler = None
            outliers = None
        else:
            scaler = request.POST["scalers"]
            outliers = request.POST["outliers"]

        model_name = request.session["model_name"]

        y_pred: np.ndarray
        trained_model, state, accuracy, y_pred, outliers = models.model_train(
            model_class, y_variable, df, y_class, scaler, outliers
        )

        parameters = trained_model.get_params()
        saved_models.objects.create(
            model_name=model_name,
            model_class=model_class,
            parameters=parameters,
            y_variable=y_variable,
            y_predictions=y_pred.tolist(),
            dropped_cols=dropped_cols,
            random_state=state,
            accuracy=float(accuracy),
            transformations="None" if scaler == None else scaler,
            outliers=outliers,
            file_trained_on=file,
            pickle_file=trained_model,
            user=request.user 
        )

        # request.session["model_name"]
        # request.session["y_class"]
        # request.session["y_variable"]
        # request.session["columns_to_drop"]
        # request.session["model_class"]

        return render(request, "train_model.html")
    else:
        return redirect("/redirect/", POST="POST")

@login_required
def graph_model_list(request: HttpRequest):
    try:
        model_list = get_list_or_404(
            saved_models,
            user=request.user
        )
    except Http404 as e:
        return render(request, "graph_model_list.html")

    viewing_table = pd.DataFrame()
    length = 0
    pks = []

    for model in model_list:
        pks.append(model.pk)
        columns = {
            "model_name": model.model_name,
            "model_class": model.model_class,
            "y_variable": model.y_variable,
            "dropped_cols": model.dropped_cols,
            "accuracy": model.accuracy,
            "transformations": model.transformations,
            "file_trained_on": model.file_trained_on,
            "upload_time": model.upload_time,
        }

        for column in columns:

            value = columns[column]
            if (
                type(value) != str
                and value is not None
                and column != "file_trained_on"
                and column != "upload_time"
            ):
                value = str(value)[1:-1]

            viewing_table.loc[length, column] = value

        length = length + 1

    table_html = viewing_table.to_html(
        classes="table table-striped table-bordered", index=False
    )

    request.session["pks"] = pks

    context = {"table": table_html, "graph": True}

    return render(request, "graph_model_list.html", context)

@login_required
def graph_accuracy(request: HttpRequest):

    model_indexes = [int(index) for index in request.POST.getlist("selected_models")]

    pks = request.session["pks"]

    model_pks = [pks[model_index] for model_index in model_indexes]
    data = accuracy_query(model_pks)
    if len(model_pks) > 1:
        title = "Accuracy Comparison.png"

    else:
        if "title" in request.POST:
            if request.POST["title"] != "":
                title = f"{request.POST["title"]}.png"
            else:
                title = f"{data["model_name"].loc[0]} Accuracy Comparison.png"
        else:
            title = f"{data["model_name"].loc[0]} Accuracy Comparison.png"

    # plt.figure(figsize=(15, 10))

    # request.POST["style"]
    plt.style.use("default")

    # Function to convert data from database query into dataframe

    sns.barplot(x="model_class", y="accuracy", hue="transformations", data=data)
    plt.title(
        title[:-4],
        fontdict={"size": 18, "fontweight": "bold"},
    )
    plt.xlabel("Model", fontdict={"size": 14, "fontweight": "bold"})
    plt.ylabel("Accuracy", fontdict={"size": 14, "fontweight": "bold"})
    plt.legend(loc="lower right")

    # if os.path.exists(rf"{MEDIA_ROOT}\plots\{title}.png"):
    #     i = 1
    #     title = title + f"({i})"
    #     if os.path.exists(rf"{MEDIA_ROOT}\plots\{title}.png"):
    #         while os.path.exists(rf"{MEDIA_ROOT}\plots\{title}{end}.png"):
    #             i += 1
    #             end = f"({i})"

    #         title = title + end

    #     plt.savefig(rf"{MEDIA_ROOT}\plots\{title}.png")

    # else:
    #     plt.savefig(rf"{MEDIA_ROOT}\plots\{title}.png")

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    # image_bytes = buf.getvalue()
    django_image = ImageFile(buf, name=title)
    saved_image = user_images.objects.create(image_name=title, image=django_image)
    path = saved_image.image.url

    context = {
        "path": rf"{path}",
        "title": title,
    }

    buf.close()
    plt.close()
    return render(request, "accuracy_graph.html", context)

def accuracy_query(pks):
    model_list = []
    data_df = pd.DataFrame(
        columns=["model_name", "accuracy", "transformations", "file_trained_on"]
    )
    length = 0

    for pk in pks:
        model = get_object_or_404(saved_models, pk=pk)
        model_list.append(model)

    for model in model_list:
        model: saved_models
        columns = {
            "model_name": model.model_name,
            "accuracy": model.accuracy,
            "transformations": model.transformations,
            "file_trained_on": model.file_trained_on,
            "model_class": model.model_class,
        }

        new_row = pd.DataFrame(
            [{column_name: value for column_name, value in columns.items()}]
        )

        data_df = pd.concat([data_df, new_row], ignore_index=True)

    return data_df

@login_required
def graph_confusion(request: HttpRequest):
    models = model_predicting()
    model_indexes = request.POST["selected_models"]

    y, y_pred, state, model_name = confusion_query(models, model_indexes)
    cm, cm_report = models.confusion_matrix(y, y_pred, state)

    df = pd.DataFrame(cm)
    table_html = df.to_html(classes="table table-striped table-bordered", index=False)

    context = {"table": table_html}

    return render(request, "show_table.html", context)

def confusion_query(models: model_predicting, model_indexes):
    model = get_object_or_404(saved_models, pk=int(model_indexes) + 1)
    columns = [
        "model_name",
        "y_variable",
        "y_predictions",
        "random_state",
        "outliers",
        "file_trained_on",
    ]

    file = get_object_or_404(upload_file, pk=model.file_trained_on)
    df = pd.read_csv(file.file)
    y = df[model.y_variable]

    if model.outliers is True:
        y = models.outlier_removal(y)

    return y, model.y_predictions, model.random_state, model.model_name

@login_required
def zenml_model_list(request: HttpRequest):
    try:
        model_list = get_list_or_404(
            saved_models,
            user=request.user
        )
    except Http404 as e:
        return render(request, "graph_model_list.html")

    viewing_table = pd.DataFrame()
    length = 0
    pks = []

    for model in model_list:
        pks.append(model.pk)
        columns = {
            "model_name": model.model_name,
            "model_class": model.model_class,
            "y_variable": model.y_variable,
            "dropped_cols": model.dropped_cols,
            "accuracy": model.accuracy,
            "transformations": model.transformations,
            "file_trained_on": model.file_trained_on,
            "upload_time": model.upload_time,
        }

        for column in columns:

            value = columns[column]
            if (
                type(value) != str
                and value is not None
                and column != "file_trained_on"
                and column != "upload_time"
            ):
                value = str(value)[1:-1]

            viewing_table.loc[length, column] = value

        length = length + 1

    table_html = viewing_table.to_html(
        classes="table table-striped table-bordered", index=False
    )

    request.session["pks"] = pks

    context = {"table": table_html}

    return render(request, "zenml_model_list.html", context)

@login_required
def zenml_train_pipeline(request: HttpRequest):
    # send all parameters to class for zenml to use
    # make it so you can choose parametrs from models stored in django databse to zenml
    model_indexes = [int(index) for index in request.POST.getlist("selected_models")]

    pks = request.session["pks"]

    model_pks = [pks[model_index] for model_index in model_indexes]
    pydantic_zenml_help = zenml_parse(**zenml_query(model_pks[0]))

    zenml_help = pydantic_model(zenml_data=pydantic_zenml_help)

    try:
        uri = run(pipeline="train", zenml_help=zenml_help)
        error = str(uri)
    except Exception as e:
        logging.exception(msg="an error occured")
        error = str(e)
        raise e

    return HttpResponse(error)

def zenml_query(pk):

    model = get_object_or_404(saved_models, pk=pk)

    model: saved_models
    parameters = {
        "model_name": model.model_name,
        "model_class": model.model_class,
        "y_variable": model.y_variable,
        "dropped_columns": model.dropped_cols,
        "transformations": model.transformations,
        "outliers": model.outliers,

        # "file_trained_on": model.file_trained_on.file,
        "random_state": model.random_state,
    }

    return parameters

@login_required
def zenml_logged_list(request: HttpRequest):
    if "update" in request.GET:
        if request.GET["update"] == "true":
            try:
                viewing_table, model_names, _ = get_registered_models_list()
            except Exception as e:
                logging.exception(msg="an error occured")
                raise e

            if viewing_table.empty:
                return render(request, "zenml_logged_model_list.html")

            table_html = viewing_table.to_html(
                classes="table table-striped table-bordered", index=False
            )

            request.session["model_names"] = model_names

            context = {"table": table_html, "logged": True}

            return render(request, "zenml_registered_model_list.html", context)

    try:
        viewing_table, uri_list, run_names = get_logged_models_list()
    except Exception as e:
        logging.exception(msg="an error occured")
        error = str(e)
        raise e

    if uri_list == None:
        return render(request, "zenml_logged_model_list.html")

    table_html = viewing_table.to_html(
        classes="table table-striped table-bordered", index=False
    )

    request.session["uri_list"] = uri_list
    request.session["run_names"] = run_names

    context = {"table": table_html}

    if request.method == "POST":
        model_names = request.session["model_names"]
        model_indexes = [
            int(index) for index in request.POST.getlist("selected_models")
        ]
        model_names = [model_names[model_index] for model_index in model_indexes]
        model_name = model_names[0]
        context["model_name"] = model_name

    return render(request, "zenml_logged_model_list.html", context)

@login_required
def get_logged_models_list():
    # Get the active MLflow experiment tracker from ZenML
    viewing_df = pd.DataFrame()
    length = 0
    uri_list = []
    run_names = []

    client = Client()
    experiment_tracker: MLFlowExperimentTracker
    experiment_tracker = client.active_stack.experiment_tracker

    # Get the MLflow tracking URI
    # tracking_uri = experiment_tracker.get_tracking_uri()
    # print(tracking_uri)
    mlflow.set_tracking_uri(os.environ.get("POSTGRES_LOCALHOSTPORT"))

    # Get experiment by name or ID
    experiment_name = "train_pipeline"
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment:
        experiment_id = experiment.experiment_id

        # Search for all runs in the experiment
        runs = mlflow.search_runs(experiment_ids=[experiment_id])
        runs = pd.DataFrame(runs)
        # Filter runs that have logged models

        logging.warning(f"Found {len(runs)} runs with logged models")

        # Get detailed model information from each run

        for idx, run in runs.iterrows():
            run_id = run["run_id"]
            uri = run["artifact_uri"]
            uri_list.append(uri)
            run_name = run.get("tags.mlflow.runName", "No name set")
            run_names.append(run_name)

            start_timestamp: pd.Timestamp

            start_timestamp = run["start_time"]
            start_time = start_timestamp.to_pydatetime().replace(
                microsecond=0, tzinfo=None
            )

            viewing_df.loc[length, "run_name"] = run_name
            viewing_df.loc[length, "date"] = start_time
            length = length + 1

        return viewing_df, uri_list, run_names

    else:
        return None, None, None

@login_required
def zenml_register_pipeline(request: HttpRequest):
    model_indexes = [int(index) for index in request.POST.getlist("selected_models")]

    run_names = request.session["run_names"]
    model_run_names = [run_names[model_index] for model_index in model_indexes]
    run_name = model_run_names[0]

    uri_list = request.session["uri_list"]

    model_uris = [uri_list[model_index] for model_index in model_indexes]

    uri = model_uris[0]

    pydantic_zenml_help = zenml_parse(
        model_name=(
            request.POST["model_name"] if request.POST["model_name"] != "" else run_name
        ),
        uri=uri,
        run_name=run_name,
        registered_model_name=(
            request.POST["registered_model"]
            if "registered_model" in request.POST
            else "None"
        ),
    )
    zenml_help = pydantic_model(zenml_data=pydantic_zenml_help)

    try:
        run(pipeline="register", zenml_help=zenml_help)

    except Exception as e:
        logging.exception(msg="an error occured")
        error = str(e)
        raise e

    return HttpResponse()

@login_required
def zenml_register_list(request: HttpRequest):
    from zenml.client import Client

    # Get the root path of the active artifact store
    root_path = Client().active_stack.artifact_store.path

    logging.warning(f"The artifact store root path is: {root_path}")

    try:
        viewing_table, registered_model_names, zenml_run_names = (
            get_registered_models_list()
        )
    except Exception as e:
        logging.exception(msg="an error occured")
        raise e

    if viewing_table.empty:
        return render(request, "zenml_registered_model_list.html")

    table_html = viewing_table.to_html(
        classes="table table-striped table-bordered", index=False
    )

    request.session["registered_model_names"] = registered_model_names
    request.session["zenml_run_names"] = zenml_run_names

    context = {"table": table_html}

    return render(request, "zenml_registered_model_list.html", context)

def get_registered_models_list():
    client = Client()
    model_registry: MLFlowModelRegistry
    model_registry = client.active_stack.model_registry
    model_list = model_registry.list_models()
    mlflow_client = MlflowClient()

    viewing_df = pd.DataFrame()
    length = 0
    model_names = []
    zenml_run_names = []

    if len(model_list) > 0:
        for model in model_list:
            viewing_df.loc[length, "model_name"] = model.name

            try:
                version = model_registry.get_latest_model_version(model.name).version
                run_name = mlflow_client.get_model_version(model.name, version).tags[
                    "zenml_run_name"
                ]
                viewing_df.loc[length, "version"] = version
                viewing_df.loc[length, "run_name"] = run_name
                zenml_run_names.append(run_name)

            except AttributeError as e:
                print(f"{e}")
                viewing_df.loc[length, "version"] = "0"
                viewing_df.loc[length, "run_name"] = model.name
                zenml_run_names.append(model.name)

            try:
                viewing_df.loc[length, "stage"] = (
                    model_registry.get_latest_model_version(model.name).stage.name
                )
            except AttributeError as e:
                viewing_df.loc[length, "stage"] = "N/A"

            model_names.append(model.name)
            length = length + 1

    return viewing_df, model_names, zenml_run_names

@login_required
def zenml_deploy_pipeline(request: HttpRequest):

    model_indexes = [int(index) for index in request.POST.getlist("selected_models")]
    registered_model_names = request.session["registered_model_names"]
    zenml_run_names = request.session["zenml_run_names"]
    registered_model_name = [
        registered_model_names[model_index] for model_index in model_indexes
    ][0]
    zenml_run_name = [zenml_run_names[model_index] for model_index in model_indexes][0]

    pydantic_zenml_help = zenml_parse(
        registered_model_name=registered_model_name, run_name=zenml_run_name
    )
    zenml_help = pydantic_model(zenml_data=pydantic_zenml_help)

    if "deployment" in request.POST:
        deployment_name = request.POST["deployment"]
    else:
        if request.POST["model_name"] != "":
            deployment_name = request.POST["model_name"]
        else:
            deployment_name = zenml_run_name

    try:
        run(
            pipeline="deploy",
            zenml_help=zenml_help,
            deployment_name=deployment_name,
        )

    except Exception as e:
        logging.exception(msg="an error occured")
        error = str(e)
        raise e

    return HttpResponse()

@login_required
def zenml_deploy_list(request: HttpRequest):

    if request.method == "POST":
        if "deployed" in request.POST:
            model_indexes = [
                int(index) for index in request.POST.getlist("selected_models")
            ]
            deployment_names = request.session["deployment_names"]
            names = [deployment_names[model_index] for model_index in model_indexes]
            deployment_name = names[0]

            try:
                viewing_table, registered_model_names, zenml_run_names = (
                    get_registered_models_list()
                )
            except Exception as e:
                logging.exception(msg="an error occured")
                raise e

            if viewing_table.empty:
                return render(request, "zenml_registered_model_list.html")

            table_html = viewing_table.to_html(
                classes="table table-striped table-bordered", index=False
            )

            request.session["registered_model_names"] = registered_model_names
            request.session["zenml_run_names"] = zenml_run_names

            context = {"table": table_html, "deployment": deployment_name}

            return render(request, "zenml_registered_model_list.html", context)

    else:
        client = Client()
        model_registry: MLFlowModelRegistry
        model_registry = client.active_stack.model_registry

        deployments = client.list_deployments()

        viewing_df = pd.DataFrame()
        length = 0
        deployment_names = []
        deployment_urls = []

        for deployment in deployments:

            viewing_df.loc[length, "Pipeline_name"] = deployment.pipeline.name
            viewing_df.loc[length, "Deployment name"] = deployment.name

            registered_model_name = (
                deployment.snapshot.pipeline_configuration.init_hook_kwargs[
                    "registered_model_name"
                ]
            )

            viewing_df.loc[length, "Registered_model_name"] = registered_model_name

            try:
                version = model_registry.get_latest_model_version(
                    registered_model_name
                ).version

                viewing_df.loc[length, "version"] = version

            except AttributeError as e:
                print(f"{e}")
                viewing_df.loc[length, "version"] = "0"

            viewing_df.loc[length, "Status"] = deployment.status

            # viewing_df[length, "version"] = deployment.config.model_version
            length = length + 1
            deployment_names.append(deployment.name)
            deployment_urls.append(deployment.url)
            # deployed_pipeline_names.append(deployment.config.pipeline_name)

        request.session["deployment_names"] = deployment_names
        request.session["deployment_urls"] = deployment_urls

        table_html = viewing_df.to_html(
            classes="table table-striped table-bordered", index=False
        )

        context = {"table": table_html}

        return render(request, "zenml_deployed_model_list.html", context)

@login_required
def upload_prediction_csv(request: HttpRequest):
    model_indexes = [int(index) for index in request.POST.getlist("selected_models")]
    deployment_names = request.session["deployment_names"]
    deployment_urls = request.session["deployment_urls"]

    deployment_name = [deployment_names[model_index] for model_index in model_indexes][
        0
    ]
    deployment_url = [deployment_urls[model_index] for model_index in model_indexes][0]

    request.session["deployment_name"] = deployment_name
    request.session["deployment_url"] = deployment_url

    return render(request, "zenml_prediction_csv_upload.html")

@login_required
def invoke_deployment(request: HttpRequest):
    deployment_url = request.session["deployment_url"]
    invoke_url = f"{deployment_url}/invoke"

    user_file = request.FILES["file"]
    pred_df = pd.read_csv(user_file)
    pred_df = pred_df.drop(
        [
            "Formulation_Number",
            "Main_Formulation_Number",
            "Sub_Formulation_Number",
            "Viscosity",
        ],
        axis=1,
    )

    predictions = []

    for index, row in pred_df.iterrows():
        row_dict = row.to_dict()
        row_json = json.dumps({"parameters": row_dict})
        response = requests.post(
            invoke_url,
            data=row_json,
            headers={"Content-Type": "application/json"},
        )
        response.raise_for_status()
        prediction = response.json()["outputs"]["output"][0]
        predictions.append(prediction)

    with tempfile.NamedTemporaryFile(
        mode="w+b", delete=True, suffix=".csv", newline=""
    ) as temp_file:
        for chunk in user_file.chunks():
            temp_file.write(chunk)

        temp_file.flush()

    return HttpResponse

@login_required
def upload_batch_csv(request: HttpRequest):
    model_indexes = [int(index) for index in request.POST.getlist("selected_models")]
    zenml_run_names = request.session["zenml_run_names"]
    zenml_run_name = [zenml_run_names[model_index] for model_index in model_indexes][0]
    request.session["zenml_run_name"] = zenml_run_name

    return render(request, "zenml_inference_csv_upload.html")

@login_required
def batch_inference(request: HttpRequest):
    user_file = request.FILES["file"]
    pred_df = pd.read_csv(user_file)

    zenml_run_name = request.session["zenml_run_name"]

    pred_df = pred_df.drop(
        [
            "Formulation_Number",
            "Main_Formulation_Number",
            "Sub_Formulation_Number",
            "Viscosity",
        ],
        axis=1,
    )
    pydantic_zenml_help = zenml_parse(run_name=zenml_run_name)
    zenml_help = pydantic_model(zenml_data=pydantic_zenml_help)

    with tempfile.NamedTemporaryFile(
        mode="w+", delete=True, suffix=".csv"
    ) as temp_file:
        pred_df.to_csv(temp_file.name, index=False)

        run(pipeline="batch", zenml_help=zenml_help, file_path=temp_file.name)

    return HttpResponse


# def graph_residuals(request : HttpRequest):
#         model_indexes = request.POST["selected_models"]

#         residuals, feature_names, model_names = residual_query(model_indexes)

#         plt.figure()
#         plt.style.use("classic")
#         colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#         plt.style.use("default")
#         plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)
#         plt.rcParams["axes.spines.right"] = False
#         plt.rcParams["axes.spines.top"] = False

#         for i,names in enumerate(feature_names):
#             title = f"Residuals for {names}"

#             plt.figure(i)

#             plt.hist(residuals[names], edgecolor="black", linewidth=0.5, stacked=True)
#             plt.tick_params(colors="#4d504f")
#             plt.legend(labels=model_names, reverse=True, fontsize=10, loc="right", shadow=True)
#             plt.title(title, fontdict={"size": 15, "weight": "bold"}, loc="center", pad=0)
#             plt.xlabel(
#                 "Residual Value",
#                 fontdict={"fontname": "Times New Roman", "size": 14, "fontweight": "bold"},
#             )
#             plt.ylabel(
#                 "Count of Residuals",
#                 fontdict={"fontname": "Times New Roman", "size": 14, "fontweight": "bold"},
#             )

#             self.save_model(title)

# def residual_query(model_indexes):
#     model_list = []
#     data_df = pd.DataFrame()
#     length = 0

#     for index in model_indexes:
#         model = get_object_or_404(saved_models, pk = int(index) + 1)
#         model_list.append(model)

#     columns = {
#         "model_name": model.model_name,
#         "y_variable": model.y_variable,
#         "y_predictions": model.y_predictions,
#         "random state": model.random_state,
#         "transformations": model.transformations,
#         "file_trained_on": model.file_trained_on
#     }

#     residuals = np.subtract(y_test, y_pred)
#     return residuals
