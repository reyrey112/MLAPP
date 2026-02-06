#!/bin/bash

python manage.py migrate --noinput
python manage.py collectstatic --noinput

exec uvicorn MLapp.asgi:application --host 0.0.0.0