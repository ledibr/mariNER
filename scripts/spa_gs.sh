#!/bin/bash
#SBATCH --job-name=spanish_gs
#SBATCH --output=logs/spanish_gs_%j.log
#SBATCH --error=logs/spanish_gs_%j.log
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=18:00:00

set -eu
ENV_NAME=cosi124a-3.9
source $HOME/envs/$ENV_NAME/bin/activate

cd $HOME/dancer-ner
# The -u flag makes sure standard output is written to the log file immediately
python -u run.py spa -m search