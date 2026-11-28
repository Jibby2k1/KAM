#!/bin/bash
#SBATCH --job-name=kam-gsm8k-formats
#SBATCH --account=bala1s
#SBATCH --qos=bala1s
#SBATCH --partition=hpg-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=00:12:00
#SBATCH --array=0-6%5
set -euo pipefail
module load pytorch/2.8.0
root=$1
work=$2
test -r "$work/eval_kam_pretrained.py"
export PYTHONPATH="$work:$root/runtime:$root/repo:$root"
export OMP_NUM_THREADS=4
python "$work/kam_gsm8k_format_effects.py" evaluate "$root" "$work" "$SLURM_ARRAY_TASK_ID"
