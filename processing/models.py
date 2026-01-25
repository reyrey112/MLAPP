from django.db import models
import inspect
from sklearn import linear_model
from picklefield import PickledObjectField

classes = [name for name, obj in inspect.getmembers(linear_model, inspect.isclass)]


# Create your models here.
class upload_file(models.Model):
    file = models.FileField(upload_to="csvs/")
    upload_time = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey("auth.user", on_delete=models.CASCADE, null=True)
    guest = models.TextField(null=True)


class saved_models(models.Model):
    model_name = models.TextField()
    model_class = models.TextField(null=True)
    parameters = models.JSONField()
    y_variable = models.TextField()
    y_predictions = models.JSONField(null=True)
    dropped_cols = models.JSONField(null=True)
    random_state = models.IntegerField(null=True)
    accuracy = models.FloatField(null=True)
    transformations = models.JSONField(null=True)
    outliers = models.BooleanField(null=True)
    file_trained_on = models.ForeignKey(upload_file, on_delete=models.PROTECT)
    pickle_file = PickledObjectField()
    user = models.ForeignKey("auth.user", on_delete=models.CASCADE, null=True)
    upload_time = models.DateTimeField(auto_now_add=True)
    guest = models.TextField(null=True)


class user_images(models.Model):
    image_name = models.TextField()
    image = models.ImageField(upload_to="plots/")
    user = models.ForeignKey("auth.user", on_delete=models.CASCADE, null=True)
    guest = models.TextField(null=True)
