from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views


urlpatterns = [
    path("guest/", views.register_as_guest, name="guest"),
    path("register/", views.create_user, name="register"),
    # path("login/", views.logon, name="login"),
    path("redirect/", views.redirect_end, name="redirect"),
    path("", views.redirect_home, name=""),
    path("home/", views.home, name="home"),
    path("zenml/", views.zenml_home, name="zenml_home"),
    path("upload/", views.file_upload, name="file_upload"),
    path("load/", views.load_file, name="load_file"),
    path("upload/tables/", views.show_table, name="show_table"),
    path("cleaning/columns/", views.x_variable_selection, name="x_variable_selection"),
    path(
        "cleaning/y_variable/", views.y_variable_selection, name="y_variable_selection"
    ),
    path("cleaning/model_type/", views.model_selection, name="model_selection"),
    path("cleaning/scaling/", views.scaler_selection, name="scaler_selection"),
    path("training/", views.train_model, name="train_model"),
    path("graph/", views.graph_model_list, name="graph_model_list"),
    path("graph/show/", views.graph_accuracy, name="show_graph"),
    path("zenml/list/train/", views.zenml_model_list, name="zenml_model_list"),
    path(
        "zenml/pipeline/train/", views.zenml_train_pipeline, name="zenml_train_pipeline"
    ),
    path("zenml/list/logged/", views.zenml_logged_list, name="zenml_logged_list"),
    path(
        "zenml/pipeline/register/",
        views.zenml_register_pipeline,
        name="zenml_register_pipeline",
    ),
    path("zenml/list/register/", views.zenml_register_list, name="zenml_register_list"),
    path(
        "zenml/pipeline/deploy/",
        views.zenml_deploy_pipeline,
        name="zenml_deploy_pipeline",
    ),
    path("zenml/list/deploy/", views.zenml_deploy_list, name="zenml_deploy_list"),
    path("zenml/predictions/upload", views.upload_prediction_csv, name="make_predictions"),
    path("zenml/predictions/invoke", views.invoke_deployment, name="invoke_deployment"),
    path("zenml/inference/upload", views.upload_batch_csv, name="batch_predictions"),
    path("zenml/inference/predictions", views.batch_inference, name="zenml_batch_pipeline"),
    path(
        "login/",
        auth_views.LoginView.as_view(
            template_name="login.html", redirect_authenticated_user=True
        ),
        name="login",
    ),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("register/", views.create_user, name="register"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
