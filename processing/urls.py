from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth import views as auth_views


urlpatterns = [
    path("redirect/", views.redirect_end, name="redirect"),
    path("", views.redirect_home, name=""),
    path("home/", views.home, name="home"),
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
