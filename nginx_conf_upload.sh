#!/bin/bash

envsubst '${DJANGO_PORT} ${MLFLOW_PORT} ${ZENML_PORT} ${SERVER_NAME} ${SSL_CERTIFICATE} ${SSL_CERTIFICATE_KEY}' \
    < /home/ubuntu/MLAPP/nginx.conf.template \
    > /etc/nginx/nginx.conf

nginx -t && systemctl restart nginx