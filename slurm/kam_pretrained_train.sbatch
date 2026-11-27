#!/bin/bash
#SBATCH --job-name=kam-pretrained-fit
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:35:00
#SBATCH --array=0-1
set -euo pipefail
module load pytorch/2.8.0
root=$1
export PYTHONPATH="$root/runtime:$root/repo"
export OMP_NUM_THREADS=4
arms=(kam dot)
cd "$root"
python train_kam_pretrained.py "$root" --arm "${arms[$SLURM_ARRAY_TASK_ID]}"
