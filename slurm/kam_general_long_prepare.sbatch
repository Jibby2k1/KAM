#!/bin/bash
#SBATCH --job-name=kam-long-data
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
set -euo pipefail
module load pytorch/2.8.0
root=$1
export PYTHONPATH="$root/runtime:$root/repo"
python "$root/prepare_kam_long_corpus.py" "$root"
python "$root/train_kam_general_long.py" "$root"
