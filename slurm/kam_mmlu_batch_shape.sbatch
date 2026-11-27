#!/bin/bash
#SBATCH --job-name=kam-mmlu-shape
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:05:00
#SBATCH --array=0-2
set -euo pipefail
module load pytorch/2.8.0
root=$1
export PYTHONPATH="$root/runtime:$root/repo:$root"
export OMP_NUM_THREADS=4
python "$root/probe_kam_mmlu_batch_shape.py" "$root" "$SLURM_ARRAY_TASK_ID"
