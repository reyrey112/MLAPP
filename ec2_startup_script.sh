#!/bin/bash

sudo apt update && sudo apt upgrade -y
sudo apt install gnome-terminal
sudo snap install docker
sudo apt install git -y

if [ -d "/home/ubuntu/MLAPP" ]; then
    cd /home/ubuntu/MLAPP && git pull
else
    git clone https://github.com/reyrey112/MLAPP /home/ubuntu/MLAPP
    cd /home/ubuntu/MLAPP
fi


sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13 python3.13-venv python3.13-dev
sudo apt install -y certbot python3-certbot-nginx

if [ ! -d "venv_MLAPP" ]; then
    python3.13 -m venv venv_MLAPP
fi
source venv_MLAPP/bin/activate
pip install uv

sudo apt update && sudo apt install -y --no-install-recommends 'build-essential'
uv pip install -r requirements.txt

mkdir -p ~/.aws
if [ ! -f ~/.aws/config ]; then
    echo "[default]
region = us-east-2" > ~/.aws/config
fi

python3.13 download_from_ssm.py

sudo apt update
sudo apt install nginx

set -a
source /home/ubuntu/MLAPP/.env
set +a

export SSL_CERTIFICATE=/etc/letsencrypt/live/${SERVER_NAME}/fullchain.pem
export SSL_CERTIFICATE_KEY=/etc/letsencrypt/live/${SERVER_NAME}/privkey.pem


envsubst '${DJANGO_PORT} ${MLFLOW_PORT} ${ZENML_PORT} ${SERVER_NAME} ${SSL_CERTIFICATE} ${SSL_CERTIFICATE_KEY}' \
    < /home/ubuntu/MLAPP/nginx.conf.template \
    > /etc/nginx/nginx.conf

nginx -t && systemctl restart nginx

sudo snap install aws-cli --classic
sudo aws ecr get-login-password --region us-east-2 | sudo docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-2.amazonaws.com

sudo docker-compose pull
sudo docker image prune --force
sudo docker-compose -f docker-compose.yaml up -d
