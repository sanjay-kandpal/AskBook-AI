#!/bin/bash
echo "Starting up"
python -m pip install --upgrade pip
pip install -r requirements.txt
gunicorn --bind=0.0.0.0 --timeout 600 app:app