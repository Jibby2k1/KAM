#!/bin/bash
#SBATCH --job-name=kam-preserve-eval
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:14:00
#SBATCH --array=0-18
set -euo pipefail
module load pytorch/2.8.0
root=$1
export PYTHONPATH="$root/runtime:$root/repo"
export OMP_NUM_THREADS=4
python "$root/eval_kam_adapter_preservation.py" "$root" "$SLURM_ARRAY_TASK_ID"
