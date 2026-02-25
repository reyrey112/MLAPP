#!/bin/bash

python manage.py migrate --noinput
python manage.py collectstatic --noinput

zenml init



exec gunicorn MLapp.asgi:application -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000