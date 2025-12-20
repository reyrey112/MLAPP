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

matplotlib.use("agg")
import matplotlib.pyplot as plt
import seaborn as sns
from MLapp.settings import MEDIA_URL, MEDIA_ROOT
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.http import Http404

import os, io, uuid

# import seaborn as sns

# data = pd.read_csv(r"C:\Users\reyde\Desktop\Formulations.csv")


# plot = sns.histplot(data=data, y="Viscosity")


# Create your views here.


def generate_guest_id():
    return str(uuid.uuid4())


def create_user(request: HttpRequest):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect(
                "/login/"
            )  # Redirect to a login page or another appropriate page
    else:
        form = UserCreationForm()
    return render(request, "register.html", {"form": form})


def redirect_home(request: HttpRequest):
    return redirect("/home/")




def redirect_end(request: HttpRequest, **kwargs):
    return render(request, "redirect.html", context=kwargs)


def home(request: HttpRequest):
    if request.user.is_authenticated:
        return render(request, "home.html")
    elif "guest_id" in request.session:
        return render(request, "home.html")
    else:
        request.session["guest_id"] = generate_guest_id()
        return render(request, "home.html")


def file_upload(request: HttpRequest):
    return render(request, "file_upload.html")


def load_file(request: HttpRequest):
    
    try:
        file_list = get_list_or_404(
            upload_file,
            user=request.user if request.user.is_authenticated else None,
            guest=None if request.user.is_authenticated else request.session["guest_id"],
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


def show_table(request: HttpRequest):
    if request.method == "POST":
        if "file" in request.FILES:
            try:
                user_file = request.FILES["file"]
                df = pd.read_csv(user_file)

                # Save to database
                saved_file = upload_file.objects.create(
                    file=user_file,
                    user=request.user if request.user.is_authenticated else None,
                    guest=(
                        None
                        if request.user.is_authenticated
                        else request.session["guest_id"]
                    ),
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

                df = pd.read_csv(file.file.path)

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
            get_object_or_404(upload_file, pk=request.session["dk"]).file.path
        )

        table_html = df.to_html(
            classes="table table-striped table-bordered", index=False
        )

        context = {"table": table_html}

        return render(request, "show_table.html", context)

    else:

        return redirect("/redirect/", pk="pk")


def x_variable_selection(request: HttpRequest):
    if request.method == "GET":
        df = pd.read_csv(
            get_object_or_404(upload_file, pk=request.session["dk"]).file.path
        )

        context = {"options": df.columns.values}

        return render(request, "x_variable_selection.html", context)

    else:
        return redirect("/redirect/", pk="pk")


def y_variable_selection(request: HttpRequest):
    if request.method == "POST":
        df = pd.read_csv(
            get_object_or_404(upload_file, pk=request.session["dk"]).file.path
        )

        columns_to_drop = request.POST.getlist("columns")
        df = df.drop(columns_to_drop, axis=1)

        request.session["columns_to_drop"] = columns_to_drop

        context = {"options": df.columns.values}

        return render(request, "y_variable_selection.html", context)

    else:
        return redirect("/redirect/", POST="POST")


def model_selection(request: HttpRequest):
    if request.method == "POST":
        df = pd.read_csv(
            get_object_or_404(upload_file, pk=request.session["dk"]).file.path
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


def scaler_selection(request: HttpRequest):
    if request.method == "POST":
        file = get_object_or_404(upload_file, pk=request.session["dk"])
        df = pd.read_csv(file.file.path)
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


def train_model(request: HttpRequest):
    if request.method == "POST":
        models = model_predicting()

        file = get_object_or_404(upload_file, pk=request.session["dk"])
        dropped_cols = request.session["columns_to_drop"]
        df = pd.read_csv(file.file.path)
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
            user=request.user if request.user.is_authenticated else None,
            guest=(
                None if request.user.is_authenticated else request.session["guest_id"]
            ),
        )

        # request.session["model_name"]
        # request.session["y_class"]
        # request.session["y_variable"]
        # request.session["columns_to_drop"]
        # request.session["model_class"]

        return render(request, "train_model.html")
    else:
        return redirect("/redirect/", POST="POST")


def graph_model_list(request: HttpRequest):
    try:
        model_list = get_list_or_404(
            saved_models,
            user=request.user if request.user.is_authenticated else None,
            guest=None if request.user.is_authenticated else request.session["guest_id"],
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
    df = pd.read_csv(file.file.path)
    y = df[model.y_variable]

    if model.outliers is True:
        y = models.outlier_removal(y)

    return y, model.y_predictions, model.random_state, model.model_name


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
