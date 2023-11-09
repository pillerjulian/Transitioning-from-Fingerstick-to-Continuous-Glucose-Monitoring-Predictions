#!/bin/bash

#SBATCH --mail-user=daniel.kerber@students.unibe.ch
#SBATCH --mail-type=start, end, fail
#SBATCH --time=24:00:00 
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1 
#SBATCH --job-name="Diabetes test"

module load Anaconda3 ## load available module, show modules -> module avail
eval "$(conda shell.bash hook)" ## init anaconda in shell
conda activate diabetesProject ## activate environment


## Run
sbatch --cpus-per-task=20 --mail-user=daniel.kerber@students.unibe.ch --mail-type=BEGIN,FAIL,END --job-name="Diabetes test" --mem-per-cpu=32G --cpus-per-task=1 --wrap="python3 src/main.py"