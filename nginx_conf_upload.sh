#!/bin/bash
set -a
source /home/ubuntu/MLAPP/.env
set +a

envsubst '${DJANGO_PORT} ${MLFLOW_PORT} ${ZENML_PORT} ${SERVER_NAME} ${SSL_CERTIFICATE} ${SSL_CERTIFICATE_KEY}' \
    < /home/ubuntu/MLAPP/nginx.conf.template \
    > /etc/nginx/nginx.conf

nginx -t && systemctl reload nginx