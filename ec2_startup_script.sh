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

TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

echo "SERVER_NAME=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" \
  http://169.254.169.254/latest/meta-data/public-ipv4)" | sudo tee -a /home/ubuntu/MLAPP/.env

set -a
source /home/ubuntu/MLAPP/.env
set +a


if [ ! -f "nginx-selfsigned.crt" ] || [ ! -f "nginx-selfsigned.key" ]; then
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout nginx-selfsigned.key -out nginx-selfsigned.crt \
        -subj "/C=US/ST=New Jersey/L=East Brunswick/O=Insight4Data/OU=ENG/CN=${SERVER_NAME}"
fi

export SSL_CERTIFICATE=/home/ubuntu/MLAPP/nginx-selfsigned.crt
export SSL_CERTIFICATE_KEY=/home/ubuntu/MLAPP/nginx-selfsigned.key


envsubst '${DJANGO_PORT} ${MLFLOW_PORT} ${ZENML_PORT} ${SERVER_NAME} ${SSL_CERTIFICATE} ${SSL_CERTIFICATE_KEY}' \
    < /home/ubuntu/MLAPP/nginx.conf.template \
    > /etc/nginx/nginx.conf

nginx -t && systemctl restart nginx

sudo snap install aws-cli --classic
sudo aws ecr get-login-password --region us-east-2 | sudo docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-2.amazonaws.com

sudo docker-compose pull

sudo docker-compose -f home/ubuntu/MLAPP/docker-compose.yaml up -d
