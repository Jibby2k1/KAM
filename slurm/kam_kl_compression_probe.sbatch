#!/bin/bash
#SBATCH --job-name=kam-kl-compression
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:07:00
#SBATCH --array=0-1
set -euo pipefail
module load pytorch/2.8.0
root=$1
export PYTHONPATH="$root/runtime:$root/repo:$root"
export OMP_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
arms=(kam dot)
python "$root/probe_kam_kl_compression.py" "$root" --arm "${arms[$SLURM_ARRAY_TASK_ID]}"
