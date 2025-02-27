#!/usr/bin/env bash
# exit on error
set -o errexit

# Update and install system dependencies
apt-get update
apt-get install -y python3-pip python3-dev

# Upgrade pip and setuptools
python3 -m pip install --upgrade pip setuptools wheel

# Install requirements
python3 -m pip install -r requirements.txt

# Verify gunicorn installation
python3 -m pip show gunicorn
