#!/bin/bash

# Make sure the script fails on error
set -e

# Create necessary directories
mkdir -p uploads data vectorstore/db_faiss

# Start Gunicorn
gunicorn --bind=0.0.0.0:8000 app:app