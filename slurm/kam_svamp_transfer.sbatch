#!/bin/bash
#SBATCH --job-name=kam-svamp
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:08:00
#SBATCH --array=2,3,8,9,14,15,24
set -euo pipefail
module load pytorch/2.8.0
root=$1
shift
export PYTHONPATH="$root/runtime:$root/repo:$root"
export OMP_NUM_THREADS=4
python "$root/eval_kam_svamp_transfer.py" "$root" "$SLURM_ARRAY_TASK_ID" "$@"
