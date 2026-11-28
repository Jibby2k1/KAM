#!/bin/bash
#SBATCH --job-name=kam-tabmwp-archive
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:08:00
#SBATCH --array=0-6%5
set -euo pipefail
module load pytorch/2.8.0
root=$1
work=$2
export PYTHONPATH="$work:$root/runtime:$root/repo:$root"
export OMP_NUM_THREADS=4
python "$work/kam_tabmwp_archive_noise.py" evaluate "$root" "$work" "$SLURM_ARRAY_TASK_ID"
