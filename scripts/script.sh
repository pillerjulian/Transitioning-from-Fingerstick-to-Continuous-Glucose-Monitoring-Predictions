#!/bin/bash

#SBATCH --mail-user=daniel.kerber@students.unibe.ch
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=20
#SBATCH --job-name="Diabetes Project"

# Load the Anaconda module
module load Anaconda3
eval "$(conda shell.bash hook)"
conda activate diabetesProject

# Run your Python script
python3 src/main.py
