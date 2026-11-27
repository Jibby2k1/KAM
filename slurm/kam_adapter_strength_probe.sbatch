#!/bin/bash
#SBATCH --job-name=kam-strength
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:14:00
#SBATCH --array=0-3
set -euo pipefail
module load pytorch/2.8.0
root=$1
export PYTHONPATH="$root/runtime:$root/repo"
export OMP_NUM_THREADS=4
recipes=(reference high_lr_kl)
arms=(kam dot)
recipe=${recipes[$((SLURM_ARRAY_TASK_ID / 2))]}
arm=${arms[$((SLURM_ARRAY_TASK_ID % 2))]}
python "$root/probe_kam_adapter_strength.py" "$root" "$root/strength-$recipe-$arm.json" --recipe "$recipe" --arm "$arm"
