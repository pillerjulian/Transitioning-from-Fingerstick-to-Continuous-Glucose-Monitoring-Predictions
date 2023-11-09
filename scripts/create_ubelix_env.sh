#!/bin/bash

# Load the Anaconda module
module load Anaconda3
eval "$(conda shell.bash hook)"

# Create a new environment for MIALab
conda create --name diabetesProject python=3.10.13

# Activate the new environment
conda activate diabetesProject

# Install the MIALab dependencies
pip install -r requirements.txt