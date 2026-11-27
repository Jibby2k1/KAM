#!/bin/bash
#SBATCH --job-name=kam-matched-drift
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=00:05:00
#SBATCH --array=0,1,6,7,12,13,18-23
set -euo pipefail
module load pytorch/2.8.0
root=$1
export PYTHONPATH="$root/runtime:$root/repo:$root"
export OMP_NUM_THREADS=4
python "$root/probe_kam_matched_drift.py" "$root" "$SLURM_ARRAY_TASK_ID"
