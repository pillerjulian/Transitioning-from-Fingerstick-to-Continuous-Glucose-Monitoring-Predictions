#!/bin/bash
module load Anaconda3 ## load available module, show modules -> module avail
eval "$(conda shell.bash hook)" ## init anaconda in shell
conda activate diabetesProject ## activate environment
## Run
sbatch --mail-user=diego.zeiter1@students.unibe.ch --mail-type=BEGIN,FAIL,END --job-name="Diabetes Project" --mem-per-cpu=32G --cpus-per-task=20 --wrap="python3 src/main.py"