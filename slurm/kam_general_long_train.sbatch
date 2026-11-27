#!/bin/bash
#SBATCH --job-name=kam-long-train
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-1
set -euo pipefail
module load pytorch/2.8.0
root=$1
export OMP_NUM_THREADS=4
export PYTHONPATH="$root/runtime:$root/repo"
arm=(T-KAM-F T-WIDE)
python "$root/train_kam_general_long.py" "$root" --arm "${arm[$SLURM_ARRAY_TASK_ID]}"
