#!/bin/bash

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Check if dataset directory exists
if [ ! -d "imagenet-mini" ]; then
    echo "Downloading ImageNet Mini dataset..."
    kaggle datasets download ifigotin/imagenetmini-1000
    # Unzip the dataset
    unzip imagenetmini-1000.zip
    # Clean up zip file
    rm imagenetmini-1000.zip
fi

echo "Setup complete! Virtual environment is activated and requirements are installed."
