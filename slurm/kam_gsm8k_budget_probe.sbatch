#!/bin/bash
#SBATCH --job-name=kam-gsm8k-budget
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --array=2,3,8,9,14,15,24
set -euo pipefail
module load pytorch/2.8.0
root=$1
export PYTHONPATH="$root/runtime:$root/repo:$root"
export OMP_NUM_THREADS=4
python "$root/probe_kam_gsm8k_budget.py" "$root" "$SLURM_ARRAY_TASK_ID"
