#!/bin/bash
#SBATCH --mail-user=diego.zeiter1@students.unibe.ch
#SBATCH --mail-type=start, end, fail
#SBATCH --time=24:00:00 
#SBATCH --mem-per-cpu=1G
#SBATCH --cpus-per-task=1 
#SBATCH --job-name="Diabetes Project"
module load Anaconda3 ## load available module, show modules -> module avail
eval "$(conda shell.bash hook)" ## init anaconda in shell
conda activate diabetesProject ## activate environment
## Run
sbatch --mail-user=diego.zeiter1@students.unibe.ch --mail-type=BEGIN,FAIL,END --job-name="Diabetes Project" --mem-per-cpu=32G --time=24:00:00 --cpus-per-task=20 --wrap="python3 src/main.py"