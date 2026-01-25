from django.apps import AppConfig


class ProcessingConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'processing'

    # def ready(self):
    #     from . import initializers

    #     initializers.run_on_startup()
    #     print("Initializer Running")
