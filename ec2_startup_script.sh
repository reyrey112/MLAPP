#!/bin/bash

sudo apt update && sudo apt upgrade -y
sudo apt install gnome-terminal
sudo snap install docker
sudo apt install git -y

git clone https://github.com/reyrey112/MLAPP

cd MLAPP

sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13 python3.13-venv python3.13-dev

python3.13 -m venv venv_MLAPP
source venv_MLAPP/bin/activate
pip install uv
sudo apt update && sudo apt install -y --no-install-recommends 'build-essential'
uv pip install -r requirement.txt

mkdir -p ~/.aws 
echo "[default]
region = us-east-2" > ~/.aws/config

python3.13 download_from_ssm.py

sudo apt update
sudo apt install nginx

echo "SERVER_NAME=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)" | sudo tee -a /home/ubuntu/MLAPP/.env

set -a
source /home/ubuntu/MLAPP/.env
set +a


openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout nginx-selfsigned.key -out nginx-selfsigned.crt \
    -subj "/C=US/ST=New Jersey/L=East Brunswick/O=Insight4Data/OU=ENG/CN=${SERVER_NAME}"

export SSL_CERTIFICATE=/home/ubuntu/MLAPP/nginx-selfsigned.crt
export SSL_CERTIFICATE_KEY=/home/ubuntu/MLAPP/nginx-selfsigned.key

sudo bash nginx_conf_upload.sh

sudo snap install aws-cli --classic
sudo aws ecr get-login-password --region us-east-2 | sudo docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-2.amazonaws.com

sudo docker-compose pull

sudo docker-compose up -d
