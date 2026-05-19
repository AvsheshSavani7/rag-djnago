#!/bin/bash
set -e

cd /opt/apps/django-app

git fetch origin
git checkout vps/main
git pull origin vps/main

docker compose up -d --build

docker builder prune -af

echo "Deployment completed."
