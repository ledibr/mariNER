#!/bin/bash
#SBATCH --job-name=zulu
#SBATCH --output=zulu.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000
#SBATCH --time=18:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cosi232

cd $HOME/models/multilingual  # adjust location as needed
python3 zulu.py  # adjust name as needed
