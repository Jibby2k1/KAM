#!/bin/bash
#SBATCH --job-name=kam-eval-merge
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
set -euo pipefail
module load pytorch/2.8.0
root=$1
python "$root/merge_kam_general_generation.py" "$root"
