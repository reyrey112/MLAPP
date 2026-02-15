from django.db import models
import inspect
from sklearn import linear_model
from picklefield import PickledObjectField
from cryptography.fernet import Fernet
import pgtrigger
from django.contrib.auth.models import AbstractUser

classes = [name for name, obj in inspect.getmembers(linear_model, inspect.isclass)]


# Create your models here.


# class User_Credentials(models.Model):
#     user = models.OneToOneField(User, on_delete=models.CASCADE)
#     username = models.CharField(max_length=100)
#     password = models.TextField()  # Encrypted
#     encryption_key = models.TextField()

#     def set_encrypted_field(self, field_name: str, value: str):
#         """Encrypt and store a field"""
#         f = Fernet(self.encryption_key.encode())
#         encrypted = f.encrypt(value.encode())
#         setattr(self, field_name, encrypted.decode())

#     def get_decrypted_field(self, field_name: str) -> str:
#         """Decrypt and return a field"""
#         f = Fernet(self.encryption_key.encode())
#         encrypted = getattr(self, field_name).encode()
#         return f.decrypt(encrypted).decode()
from django.contrib.auth.models import User
from django.db import models
from django.db.models.signals import post_save
from django.dispatch import receiver

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    is_guest = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.user.username}'s profile"


@receiver(post_save, sender=User)
def create_or_update_user_profile(sender, instance, created, **kwargs):
    """Create or update user profile when user is saved"""
    if created:
        UserProfile.objects.create(user=instance)
    else:
        # Update existing profile if needed
        instance.profile.save()

class upload_file(models.Model):
    file = models.FileField(upload_to="csvs/")
    upload_time = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey("auth.user", on_delete=models.CASCADE, null=True)


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


class user_images(models.Model):
    image_name = models.TextField()
    image = models.ImageField(upload_to="plots/")
    user = models.ForeignKey("auth.user", on_delete=models.CASCADE, null=True)
